from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate bounded best-of-N oracle evasion over a fixed, ordered score panel."
        )
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--threshold-kind",
        required=True,
        choices=(
            "fixture_detector_block_threshold",
            "runtime_block_threshold",
            "runtime_review_threshold",
        ),
    )
    parser.add_argument(
        "--traffic-mode",
        default="not_applicable",
        choices=("not_applicable", "shadow", "review", "enforce"),
    )
    parser.add_argument("--query-budgets", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    detector = _detector_metadata(args.detector)
    artifact_threshold = float(detector["artifact_threshold"])
    if args.threshold_kind == "runtime_review_threshold" and args.threshold is None:
        raise ValueError(
            "--threshold is required for runtime_review_threshold; a detector artifact "
            "stores its block threshold, not its runtime review threshold"
        )
    if args.threshold_kind.startswith("runtime_") and args.traffic_mode == "not_applicable":
        raise ValueError("runtime threshold analyses require shadow, review, or enforce traffic mode")
    threshold = float(args.threshold) if args.threshold is not None else artifact_threshold
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    if (
        args.threshold_kind == "fixture_detector_block_threshold"
        and not math.isclose(threshold, artifact_threshold, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(
            "fixture_detector_block_threshold must equal the supplied detector artifact threshold"
        )
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"refusing to overwrite {args.output}; choose a versioned output or pass --force"
        )

    feature_provenance = _feature_metadata(args.features)
    with args.scores.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "base_sample_id",
        "query_index",
        "variant_type",
        "risk_score",
        "label_id",
    }
    if not rows or required - set(rows[0]):
        raise ValueError(f"scored variants must contain columns {sorted(required)}")
    if any(row["label_id"].strip() != "1" for row in rows):
        raise ValueError("bounded evasion evaluation requires a malicious-only score panel")
    recorded = [_recorded_detector_threshold(row) for row in rows]
    if any(value is None for value in recorded):
        raise ValueError(
            "score rows must record detector_threshold (or the legacy threshold alias)"
        )
    if any(
        not math.isclose(float(value), artifact_threshold, rel_tol=0.0, abs_tol=1e-12)
        for value in recorded
    ):
        raise ValueError("score-row detector threshold does not match the supplied artifact")

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        score = float(row["risk_score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("risk_score values must be in [0,1]")
        groups.setdefault(row["base_sample_id"], []).append(row)
    for sample_id, variants in groups.items():
        indices = [int(row["query_index"]) for row in variants]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate query_index values for {sample_id}")
        if sorted(indices) != list(range(1, len(indices) + 1)):
            raise ValueError(f"query_index values must be contiguous from one for {sample_id}")

    summary: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for budget in sorted(set(args.query_budgets)):
        if budget < 1:
            raise ValueError("query budgets must be positive")
        underfilled = [
            sample_id for sample_id, variants in groups.items() if len(variants) < budget
        ]
        if underfilled:
            raise ValueError(
                f"query budget {budget} exceeds the available panel for {underfilled}"
            )
        worst_scores: list[float] = []
        for sample_id, variants in sorted(groups.items()):
            eligible = sorted(
                variants,
                key=lambda row: int(row["query_index"]),
            )[:budget]
            if not eligible:
                raise ValueError(f"no variants available for {sample_id}")
            selected = min(
                eligible,
                key=lambda row: (
                    float(row["risk_score"]),
                    int(row["query_index"]),
                ),
            )
            score = float(selected["risk_score"])
            worst_scores.append(score)
            selections.append(
                {
                    "query_budget": budget,
                    "base_sample_id": sample_id,
                    "selected_variant": selected["variant_type"],
                    "selected_risk_score": score,
                    "evaded": score < threshold,
                }
            )
        detected = sum(score >= threshold for score in worst_scores)
        summary.append(
            {
                "query_budget": budget,
                "n_samples": len(worst_scores),
                "threshold": threshold,
                "malicious_recall": detected / len(worst_scores),
                "evasion_rate": 1.0 - detected / len(worst_scores),
                "mean_worst_case_score": sum(worst_scores) / len(worst_scores),
            }
        )

    payload = {
        "schema_version": 2,
        "method": "bounded_best_of_n_oracle_over_fixed_variant_order",
        "provenance": {
            "scores": _portable_path(args.scores),
            "scores_sha256": _sha256(args.scores),
            "detector": _portable_path(args.detector),
            "detector_sha256": _sha256(args.detector),
            "detector_metadata": detector,
            "features": _portable_path(args.features),
            "features_sha256": _sha256(args.features),
            "feature_provenance": feature_provenance,
            "threshold": threshold,
            "detector_artifact_threshold": artifact_threshold,
            "threshold_kind": args.threshold_kind,
            "traffic_mode": args.traffic_mode,
        },
        "interpretation": (
            "This is a detector-score oracle analysis, not a sequential adaptive attack. "
            "When traffic_mode is shadow, threshold-based blocking is counterfactual because "
            "the effective runtime action remains allow."
        ),
        "summary": summary,
        "selections": selections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def _detector_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        result: dict[str, object] = {
            "artifact_threshold": float(
                np.asarray(data["threshold"]).reshape(-1)[0]
            ),
            "feature_dim": int(np.asarray(data["weights"]).size),
        }
        if "metadata_json" in data:
            result.update(
                json.loads(str(np.asarray(data["metadata_json"]).reshape(-1)[0]))
            )
        for key in (
            "source",
            "model_family",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "preprocessing_sha256",
            "pooling",
        ):
            if key in data:
                result[key] = str(np.asarray(data[key]).reshape(-1)[0])
    return result


def _feature_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        result: dict[str, object] = {
            "rows": int(np.asarray(data["sample_ids"]).reshape(-1).size),
            "feature_source": str(
                np.asarray(data["feature_source"]).reshape(-1)[0]
            ),
        }
        for key in ("text_embeddings", "image_embeddings", "attribution_features"):
            if key in data:
                result[f"{key}_shape"] = list(np.asarray(data[key]).shape)
    return result


def _recorded_detector_threshold(row: dict[str, str]) -> float | None:
    values = [
        float(row[name])
        for name in ("detector_threshold", "threshold")
        if row.get(name, "").strip()
    ]
    if not values:
        return None
    if any(
        not math.isclose(value, values[0], rel_tol=0.0, abs_tol=1e-12)
        for value in values[1:]
    ):
        raise ValueError("score row contains conflicting threshold aliases")
    return values[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    main()
