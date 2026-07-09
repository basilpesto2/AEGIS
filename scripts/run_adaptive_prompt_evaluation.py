from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.evaluation import predict_from_threshold, run_logistic_validation_grid
from AEGIS.io import align_embeddings_with_metadata, load_embeddings, load_metadata
from AEGIS.labels import coerce_binary_labels
from AEGIS.logistic import LogisticRegressionNumpy
from AEGIS.metrics import detection_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a bounded score-query attacker that selects the lowest-scoring prompt variant."
    )
    parser.add_argument("--original-embeddings", required=True)
    parser.add_argument("--original-metadata", required=True)
    parser.add_argument("--transformed-embeddings", required=True)
    parser.add_argument("--transformed-metadata", required=True)
    parser.add_argument("--output-summary-csv", required=True)
    parser.add_argument("--output-selections-csv", required=True)
    parser.add_argument("--output-family-csv")
    parser.add_argument("--variant-column", default="robustness_variant")
    parser.add_argument("--original-id-column", default="original_sample_id")
    parser.add_argument("--family-column", default="source_subset")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="experiment_split")
    parser.add_argument("--fit-split", default="fit")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    parser.add_argument("--l2-values", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--epochs-values", type=int, nargs="+", default=[300, 600])
    parser.add_argument("--malicious-prior", type=float, default=0.5)
    args = parser.parse_args()

    original_metadata = _require_metadata(load_metadata(args.original_metadata), "original")
    transformed_metadata = _require_metadata(load_metadata(args.transformed_metadata), "transformed")
    original_x, original_metadata, original_ids = _load_aligned(
        args.original_embeddings, original_metadata
    )
    transformed_x, transformed_metadata, _ = _load_aligned(
        args.transformed_embeddings, transformed_metadata
    )
    for column in [args.label_column, args.split_column]:
        _require_column(original_metadata, column, "original metadata")
    for column in [args.variant_column, args.original_id_column]:
        _require_column(transformed_metadata, column, "transformed metadata")

    labels = coerce_binary_labels(original_metadata[args.label_column])
    splits = original_metadata[args.split_column].astype(str).to_numpy()
    fit_mask = splits == args.fit_split
    validation_mask = splits == args.validation_split
    test_mask = splits == args.test_split
    grid = run_logistic_validation_grid(
        original_x,
        labels=labels,
        fit_mask=fit_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        learning_rates=args.learning_rates,
        l2_values=args.l2_values,
        epochs_values=args.epochs_values,
        malicious_prior=args.malicious_prior,
    )
    selected = max((result.as_row() for result in grid), key=lambda row: row["validation_auprc"])
    classifier = LogisticRegressionNumpy(
        learning_rate=float(selected["learning_rate"]),
        l2=float(selected["l2"]),
        epochs=int(selected["epochs"]),
    ).fit(original_x[fit_mask], labels[fit_mask])
    threshold = float(selected["threshold"])

    original_scores = classifier.predict_proba(original_x)
    transformed_scores = classifier.predict_proba(transformed_x)
    selection_columns = ["sample_id", args.original_id_column, args.variant_column]
    if args.family_column in transformed_metadata.columns:
        selection_columns.append(args.family_column)
    transformed = transformed_metadata[selection_columns].copy()
    transformed["score"] = transformed_scores
    selected_indices = transformed.groupby(args.original_id_column)["score"].idxmin()
    adaptive = transformed.loc[selected_indices].copy().sort_values(args.original_id_column)

    id_to_index = {sample_id: index for index, sample_id in enumerate(original_ids.astype(str))}
    original_indices = np.asarray(
        [id_to_index[sample_id] for sample_id in adaptive[args.original_id_column].astype(str)]
    )
    if np.any(labels[original_indices] != 1) or np.any(~test_mask[original_indices]):
        raise ValueError("Adaptive candidates must map only to malicious original test rows.")
    baseline_malicious_scores = original_scores[original_indices]
    adaptive_scores = adaptive["score"].to_numpy(dtype=np.float64)
    benign_test_scores = original_scores[test_mask & (labels == 0)]
    benign_labels = labels[test_mask & (labels == 0)]

    summary_rows = []
    baseline_target_scores = np.concatenate([baseline_malicious_scores, benign_test_scores])
    adaptive_target_scores = np.concatenate([adaptive_scores, benign_test_scores])
    target_labels = np.concatenate(
        [np.ones(len(adaptive_scores), dtype=np.int64), benign_labels]
    )
    for condition, scores in [
        ("original", baseline_target_scores),
        ("adaptive_min_score", adaptive_target_scores),
    ]:
        predictions = predict_from_threshold(scores, threshold)
        row = {
            "condition": condition,
            "query_budget": int(
                transformed_metadata[args.variant_column].astype(str).nunique()
            ),
            "threshold": threshold,
            "mean_malicious_score": float(np.mean(scores[: len(adaptive_scores)])),
            "malicious_recall": float(np.mean(predictions[: len(adaptive_scores)] == 1)),
        }
        row.update(detection_report(target_labels, scores, predictions))
        summary_rows.append(row)

    baseline_detected = baseline_malicious_scores >= threshold
    adaptive_evaded = adaptive_scores < threshold
    adaptive["original_score"] = baseline_malicious_scores
    adaptive["score_delta"] = adaptive_scores - baseline_malicious_scores
    adaptive["original_detected"] = baseline_detected.astype(int)
    adaptive["adaptive_detected"] = (~adaptive_evaded).astype(int)
    adaptive["successful_evasion"] = (baseline_detected & adaptive_evaded).astype(int)
    evasion_rate = float(
        np.mean(adaptive_evaded[baseline_detected]) if np.any(baseline_detected) else 0.0
    )
    for row in summary_rows:
        row["adaptive_evasion_rate_among_originally_detected"] = (
            0.0 if row["condition"] == "original" else evasion_rate
        )
        row["mean_score_delta_vs_original"] = (
            0.0
            if row["condition"] == "original"
            else float(np.mean(adaptive_scores - baseline_malicious_scores))
        )

    family_rows = []
    if args.family_column in adaptive.columns:
        family_values = adaptive[args.family_column].astype(str).to_numpy()
        for family in sorted(set(family_values)):
            family_mask = family_values == family
            family_labels = np.concatenate(
                [np.ones(int(np.sum(family_mask)), dtype=np.int64), benign_labels]
            )
            for condition, malicious_scores in [
                ("original", baseline_malicious_scores[family_mask]),
                ("adaptive_min_score", adaptive_scores[family_mask]),
            ]:
                scores = np.concatenate([malicious_scores, benign_test_scores])
                predictions = predict_from_threshold(scores, threshold)
                row = {
                    "condition": condition,
                    "family": family,
                    "n_malicious": int(np.sum(family_mask)),
                    "n_benign": int(len(benign_labels)),
                    "malicious_recall": float(
                        np.mean(predictions[: int(np.sum(family_mask))] == 1)
                    ),
                }
                row.update(detection_report(family_labels, scores, predictions))
                family_rows.append(row)

    summary_path = Path(args.output_summary_csv)
    selections_path = Path(args.output_selections_csv)
    family_path = (
        Path(args.output_family_csv)
        if args.output_family_csv
        else summary_path.with_name(f"{summary_path.stem}_by_family.csv")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    selections_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    adaptive.to_csv(selections_path, index=False)
    pd.DataFrame(family_rows).to_csv(family_path, index=False)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "selections": str(selections_path),
                "family_summary": str(family_path),
                "selected_hyperparameters": selected,
                "variant_selection_counts": adaptive[args.variant_column].value_counts().to_dict(),
                "adaptive_evasion_rate": evasion_rate,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _load_aligned(path: str, metadata: pd.DataFrame):
    embeddings, sample_ids = load_embeddings(path)
    aligned, aligned_metadata, aligned_ids = align_embeddings_with_metadata(
        embeddings, sample_ids, metadata
    )
    if aligned_metadata is None:
        raise ValueError("metadata alignment failed.")
    return aligned, aligned_metadata, aligned_ids


def _require_metadata(metadata: pd.DataFrame | None, name: str) -> pd.DataFrame:
    if metadata is None:
        raise ValueError(f"{name} metadata is required.")
    return metadata


def _require_column(metadata: pd.DataFrame, column: str, name: str) -> None:
    if column not in metadata.columns:
        raise ValueError(f"{name} is missing required column {column!r}.")


if __name__ == "__main__":
    main()
