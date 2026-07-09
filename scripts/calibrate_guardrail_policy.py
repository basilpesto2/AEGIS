from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.calibration import CalibrationCriteria, calibrate_threshold, policy_payload
from AEGIS.detector_artifact import load_detector_artifact
from AEGIS.io import align_embeddings_with_metadata, load_embeddings, load_metadata
from AEGIS.labels import coerce_binary_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a guardrail policy from representative labeled traffic."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--output-policy", required=True)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--review-margin", type=float, default=0.05)
    parser.add_argument("--name", default="aegis-calibrated-policy")
    parser.add_argument("--allow-provenance-mismatch", action="store_true")
    args = parser.parse_args()

    artifact = load_detector_artifact(args.detector)
    features, sample_ids = load_embeddings(args.features)
    metadata = load_metadata(args.metadata)
    features, metadata, _ = align_embeddings_with_metadata(
        features,
        sample_ids,
        metadata,
        id_column=args.id_column,
    )
    if metadata is None:
        raise ValueError("Metadata is required for calibration.")
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

    scores = artifact.score(features)
    labels = coerce_binary_labels(metadata[args.label_column])
    calibration = calibrate_threshold(
        labels,
        scores,
        CalibrationCriteria(
            max_false_positive_rate=args.max_fpr,
            min_recall=args.min_recall,
            review_margin=args.review_margin,
        ),
    )
    policy = policy_payload(
        calibration,
        detector_path=args.detector,
        name=args.name,
        require_matching_provenance=not args.allow_provenance_mismatch,
    )

    output_policy = Path(args.output_policy)
    output_policy.parent.mkdir(parents=True, exist_ok=True)
    output_policy.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_summary:
        output_summary = Path(args.output_summary)
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        output_summary.write_text(
            json.dumps(calibration, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps({"policy": str(output_policy), "calibration": calibration}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
