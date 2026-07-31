from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from aegis_research.experiment import ExperimentConfig, run_ablation_suite  # noqa: E402
from aegis_research.io import load_feature_bundle, load_metadata  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate AEGIS feature-view detectors.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--trusted-per-class", type=int)
    parser.add_argument("--pseudo-per-class", type=int, default=0)
    parser.add_argument("--pseudo-low", type=float, default=0.15)
    parser.add_argument("--pseudo-high", type=float, default=0.85)
    parser.add_argument("--force", action="store_true", help="Replace files in a non-empty output directory.")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {args.output_dir}; "
            "choose a versioned --output-dir or use --force"
        )

    metadata = load_metadata(args.metadata)
    bundle = load_feature_bundle(args.features, metadata)
    metadata_sha256 = _sha256(args.metadata)
    features_sha256 = _sha256(args.features)
    config = ExperimentConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        seed=args.seed,
        trusted_per_class=args.trusted_per_class,
        pseudo_per_class=args.pseudo_per_class,
        pseudo_low=args.pseudo_low,
        pseudo_high=args.pseudo_high,
    )
    results, fitted = run_ablation_suite(metadata, bundle, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "feature_ablation.csv", results)
    best = max(results, key=lambda row: (float(row["validation_auprc"]), float(row["validation_auroc"])))
    selected = fitted[str(best["feature_view"])]
    selected["detector"].save(
        str(args.output_dir / "selected_detector.npz"),
        threshold=float(selected["threshold"]),
        metadata={
            "feature_view": best["feature_view"],
            "feature_source": bundle.feature_source,
            "selection_rule": "maximum validation AUPRC then AUROC",
            "metadata_sha256": metadata_sha256,
            "features_sha256": features_sha256,
        },
    )
    test_rows: list[dict[str, object]] = []
    for index, score in zip(selected["test_indices"], selected["test_scores"]):
        row = metadata[int(index)]
        test_rows.append(
            {
                "sample_id": row["sample_id"],
                "label_id": row["label_id"],
                "attack_style": row["attack_style"],
                "risk_score": float(score),
                "threshold": float(selected["threshold"]),
                "predicted_label_id": int(float(score) >= float(selected["threshold"])),
            }
        )
    _write_csv(args.output_dir / "test_predictions.csv", test_rows)
    summary = {
        "schema_version": 2,
        "config": vars(config),
        "metadata": _portable_path(args.metadata),
        "features": _portable_path(args.features),
        "feature_source": bundle.feature_source,
        "selected_feature_view": best["feature_view"],
        "selection_rule": "maximum validation AUPRC then AUROC; test metrics are not used for selection",
        "selected_test_metrics": selected["test_metrics"],
        "limitations": (
            "deterministic_smoke_fixture results verify software only"
            if bundle.feature_source.startswith("deterministic_smoke")
            else "results apply only to the recorded model, data, preprocessing, and split"
        ),
        "provenance": {
            "metadata_sha256": metadata_sha256,
            "features_sha256": features_sha256,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty result table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    main()
