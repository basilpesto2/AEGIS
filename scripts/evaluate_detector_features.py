from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.detector_artifact import load_detector_artifact
from AEGIS.guardrail import GuardrailPolicy, GuardrailRuntime, decisions_to_frame
from AEGIS.io import align_embeddings_with_metadata, load_embeddings, load_metadata
from AEGIS.labels import coerce_binary_labels
from AEGIS.metrics import detection_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved AEGIS detector on an embedding/metadata split."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-scores", default=None)
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--block-threshold", type=float, default=None)
    parser.add_argument("--review-margin", type=float, default=None)
    args = parser.parse_args()

    artifact = load_detector_artifact(args.detector)
    features, sample_ids = load_embeddings(args.features)
    metadata = load_metadata(args.metadata)
    features, metadata, aligned_ids = align_embeddings_with_metadata(
        features,
        sample_ids,
        metadata,
        id_column=args.id_column,
    )
    if metadata is None:
        raise ValueError("Metadata is required.")
    if args.label_column not in metadata:
        raise ValueError(f"Metadata is missing label column {args.label_column!r}.")
    if args.split_column or args.split:
        if not args.split_column or args.split is None:
            raise ValueError("Use --split-column and --split together.")
        if args.split_column not in metadata:
            raise ValueError(f"Metadata is missing split column {args.split_column!r}.")
        mask = metadata[args.split_column].astype(str).to_numpy() == args.split
        if not mask.any():
            raise ValueError(f"No rows found for split {args.split!r}.")
        features = features[mask]
        metadata = metadata.loc[mask].reset_index(drop=True)
        aligned_ids = aligned_ids[mask]

    threshold = (
        artifact.threshold if args.block_threshold is None else float(args.block_threshold)
    )
    review_margin = (
        artifact.uncertainty_margin if args.review_margin is None else float(args.review_margin)
    )
    scores = artifact.score(features)
    labels = coerce_binary_labels(metadata[args.label_column])
    predictions = (scores >= threshold).astype(np.int64)
    report = detection_report(labels, scores, predictions)
    false_positive_rate = (
        report["fp"] / (report["fp"] + report["tn"])
        if report["fp"] + report["tn"]
        else 0.0
    )
    false_negative_rate = (
        report["fn"] / (report["fn"] + report["tp"])
        if report["fn"] + report["tp"]
        else 0.0
    )

    runtime = GuardrailRuntime(
        artifact,
        policy=GuardrailPolicy(
            block_threshold=threshold,
            review_margin=review_margin,
            require_matching_provenance=True,
        ),
    )
    decisions = runtime.evaluate_metadata_features(
        features,
        metadata=metadata,
        sample_ids=aligned_ids,
    )
    table = decisions_to_frame(decisions)
    table[args.label_column] = metadata[args.label_column].reset_index(drop=True).to_numpy()

    summary = {
        "source": args.source or _summary_source(metadata, args.split),
        "n_samples": int(len(labels)),
        "n_benign": int(np.sum(labels == 0)),
        "n_malicious": int(np.sum(labels == 1)),
        "metrics": report,
        "accuracy": float(np.mean(predictions == labels)),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "tp": int(report["tp"]),
        "fp": int(report["fp"]),
        "fn": int(report["fn"]),
        "tn": int(report["tn"]),
        "misclassified_sample_ids": [
            str(aligned_ids[idx])
            for idx, value in enumerate(predictions != labels)
            if bool(value)
        ],
        "uncertain_sample_ids": [
            str(aligned_ids[idx])
            for idx, value in enumerate(np.abs(scores - threshold) <= review_margin)
            if bool(value)
        ],
        "detector": {
            "model_family": artifact.model_family,
            "model_id": artifact.model_id,
            "layer": artifact.layer,
            "pooling": artifact.pooling,
            "threshold": float(threshold),
            "source": artifact.source,
        },
        "features": str(Path(args.features)),
        "metadata": str(Path(args.metadata)),
        "split": args.split,
    }

    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_scores is not None:
        output_scores = Path(args.output_scores)
        output_scores.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_scores, index=False)
        summary["output_scores"] = str(output_scores)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _summary_source(metadata, split: str | None) -> str:
    if "source" not in metadata:
        return "aegis_detector_evaluation" if split is None else f"aegis_detector_evaluation_{split}"
    sources = "_".join(sorted(metadata["source"].astype(str).unique()))
    if split is None:
        return sources
    return f"{sources}_{split}"


if __name__ == "__main__":
    main()
