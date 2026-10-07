from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPOSITORY_ROOT / "deliverables" / "pipeline"
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(PIPELINE_ROOT))

from aegis_research.bordair_dual import load_detector_pair, score_pair  # noqa: E402


DUAL_MODE = "dual_or"
LEGACY_MODE = "legacy_single"
ACTION_RANK = {"allow": 0, "review": 1, "block": 2}
DUAL_SCORE_FIELDS = {
    "base_sample_id", "query_index", "variant_type", "variant_id", "label_id",
    "detector_mode", "risk_score", "detector_threshold", "review_threshold",
    "verdict", "uncertain", "recommended_action", "decisive_head",
    "text_risk_score", "text_block_threshold", "text_review_threshold",
    "text_action", "text_pooling", "image_risk_score", "image_block_threshold",
    "image_review_threshold", "image_action", "image_pooling",
    "runtime_detector_identity_sha256", "pair_identity_sha256",
}
LEGACY_SCORE_FIELDS = {
    "base_sample_id", "query_index", "variant_type", "variant_id", "risk_score",
    "label_id", "detector_threshold", "feature_view",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate bounded best-of-N oracle evasion over a fixed, ordered "
            "dual-head OR score panel. Legacy single-head evaluation requires an "
            "explicit legacy mode."
        )
    )
    parser.add_argument("--mode", required=True, choices=(DUAL_MODE, LEGACY_MODE))
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--threshold-kind",
        choices=(
            "detector_artifact_block_threshold",
            "runtime_block_threshold",
            "runtime_review_threshold",
        ),
    )
    parser.add_argument(
        "--traffic-mode", default="not_applicable",
        choices=("not_applicable", "shadow", "review", "enforce"),
    )
    parser.add_argument("--query-budgets", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _validate_mode_arguments(args)
    budgets = _query_budgets(args.query_budgets)
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"refusing to overwrite {args.output}; choose a versioned output or pass --force"
        )
    payload = (
        _evaluate_dual(args, budgets)
        if args.mode == DUAL_MODE
        else _evaluate_legacy(args, budgets)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


def _validate_mode_arguments(args: argparse.Namespace) -> None:
    if args.mode == DUAL_MODE:
        if args.pair_manifest is None:
            raise ValueError("dual_or mode requires --pair-manifest")
        forbidden = [
            name for name, value in (
                ("--detector", args.detector),
                ("--threshold", args.threshold),
                ("--threshold-kind", args.threshold_kind),
            ) if value is not None
        ]
        if forbidden:
            raise ValueError(
                "dual_or mode takes thresholds from its pair manifest; do not pass "
                + ", ".join(forbidden)
            )
    else:
        if args.detector is None or args.threshold_kind is None:
            raise ValueError(
                "legacy_single mode requires --detector and --threshold-kind"
            )
        if args.pair_manifest is not None:
            raise ValueError("legacy_single mode does not accept --pair-manifest")
        if args.threshold_kind == "runtime_review_threshold" and args.threshold is None:
            raise ValueError(
                "--threshold is required for runtime_review_threshold; a detector "
                "artifact stores its block threshold, not its runtime review threshold"
            )
    if (
        args.threshold_kind is not None
        and args.threshold_kind.startswith("runtime_")
        and args.traffic_mode == "not_applicable"
    ):
        raise ValueError(
            "runtime threshold analyses require shadow, review, or enforce traffic mode"
        )


def _evaluate_dual(
    args: argparse.Namespace, budgets: tuple[int, ...]
) -> dict[str, object]:
    pair = load_detector_pair(args.pair_manifest)
    feature_provenance, feature_ids, fused = _dual_feature_snapshot(args.features, pair)
    rows = _score_rows(args.scores, DUAL_SCORE_FIELDS)
    _validate_panel_rows(rows, feature_ids)
    recomputed = score_pair(pair, fused)
    entries = pair.manifest["artifacts"]
    evaluated: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        decision = _dual_decision(
            text_score=float(recomputed["text_scores"][index]),
            image_score=float(recomputed["image_scores"][index]),
            text_block=float(pair.text_artifact.threshold),
            text_review=float(entries["text"]["review_threshold"]),
            image_block=float(pair.image_artifact.threshold),
            image_review=float(entries["image"]["review_threshold"]),
            text_pooling=pair.text_artifact.pooling,
            image_pooling=pair.image_artifact.pooling,
        )
        _validate_dual_score_row(row, decision, pair.manifest)
        evaluated.append(
            {
                "base_sample_id": row["base_sample_id"].strip(),
                "query_index": int(row["query_index"]),
                "variant_type": row["variant_type"].strip(),
                "variant_id": row["variant_id"].strip(),
                **decision,
            }
        )

    groups = _group_panel(evaluated)
    summary: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for budget in budgets:
        _require_budget_capacity(groups, budget)
        selected_rows: list[dict[str, object]] = []
        for sample_id, variants in sorted(groups.items()):
            eligible = sorted(variants, key=lambda row: int(row["query_index"]))[:budget]
            selected = min(
                eligible,
                key=lambda row: (
                    ACTION_RANK[str(row["recommended_action"])],
                    float(row["decisive_band_progress"]),
                    int(row["query_index"]),
                ),
            )
            selected_rows.append(selected)
            head_decisions = {
                name: {
                    key: value for key, value in selected["head_decisions"][name].items()
                    if key != "action_rank"
                }
                for name in ("text", "image")
            }
            selections.append(
                {
                    "query_budget": budget,
                    "base_sample_id": sample_id,
                    "selected_variant": selected["variant_type"],
                    "selected_variant_id": selected["variant_id"],
                    "selected_query_index": selected["query_index"],
                    "recommended_action": selected["recommended_action"],
                    "decisive_head": selected["decisive_head"],
                    "risk_score": selected["risk_score"],
                    "threshold": selected["threshold"],
                    "review_threshold": selected["review_threshold"],
                    "decisive_band_progress": selected["decisive_band_progress"],
                    "head_decisions": head_decisions,
                    "evaded_block": selected["recommended_action"] != "block",
                    "fully_allowed": selected["recommended_action"] == "allow",
                }
            )
        summary.append(_dual_budget_summary(budget, selected_rows))

    artifacts = {
        name: {
            "path": _portable_path(getattr(pair.paths, name)),
            "sha256": entries[name]["sha256"],
            "pooling": entries[name]["pooling"],
            "feature_dim": entries[name]["feature_dim"],
            "block_threshold": entries[name]["block_threshold"],
            "review_threshold": entries[name]["review_threshold"],
            "source": entries[name]["source"],
        }
        for name in ("text", "image")
    }
    return {
        "schema_version": 3,
        "method": "bounded_best_of_n_oracle_over_fixed_variant_order",
        "detector_mode": "dual_or",
        "provenance": {
            "scores": _portable_path(args.scores),
            "scores_sha256": _sha256(args.scores),
            "features": _portable_path(args.features),
            "features_sha256": _sha256(args.features),
            "feature_provenance": feature_provenance,
            "pair_manifest": _portable_path(args.pair_manifest),
            "pair_manifest_sha256": _sha256(args.pair_manifest),
            "pair_identity_sha256": pair.manifest["pair_identity_sha256"],
            "runtime_detector_identity_sha256": pair.manifest[
                "runtime_detector_identity_sha256"
            ],
            "target": pair.manifest["target"],
            "composition": "or",
            "artifacts": artifacts,
            "traffic_mode": args.traffic_mode,
            "oracle_order": (
                "lowest_aggregate_action_rank_then_lowest_normalized_"
                "decisive_band_progress_then_earliest_query_index"
            ),
        },
        "interpretation": (
            "This is a detector-score oracle analysis, not a sequential adaptive "
            "attack. For each budget, it selects in hindsight the least restrictive "
            "OR-composed recommended action among the first N fixed variants. Raw "
            "scores from different heads are not compared directly; normalized "
            "within-action-band progress is used only to break action ties. Metrics "
            "describe recommended detector actions. In shadow mode, blocking and "
            "review are counterfactual because the effective runtime action remains allow."
        ),
        "summary": summary,
        "selections": selections,
    }


def _dual_budget_summary(
    budget: int, selected_rows: list[dict[str, object]]
) -> dict[str, object]:
    count = len(selected_rows)
    action_counts = {
        action: sum(row["recommended_action"] == action for row in selected_rows)
        for action in ("allow", "review", "block")
    }
    head_blocks = {
        name: sum(
            row["head_decisions"][name]["recommended_action"] == "block"
            for row in selected_rows
        )
        for name in ("text", "image")
    }
    decisive_counts = {
        name: sum(row["decisive_head"] == name for row in selected_rows)
        for name in ("text", "image")
    }
    blocked = action_counts["block"]
    intervened = blocked + action_counts["review"]
    return {
        "query_budget": budget,
        "n_samples": count,
        "malicious_recall": blocked / count,
        "evasion_rate": 1.0 - blocked / count,
        "review_or_block_recall": intervened / count,
        "allow_rate": action_counts["allow"] / count,
        "review_rate": action_counts["review"] / count,
        "block_rate": blocked / count,
        "action_counts": action_counts,
        "per_head_block_recall": {
            name: head_blocks[name] / count for name in ("text", "image")
        },
        "decisive_head_counts": decisive_counts,
        "mean_decisive_band_progress": sum(
            float(row["decisive_band_progress"]) for row in selected_rows
        ) / count,
    }


def _dual_feature_snapshot(path: Path, pair) -> tuple[dict[str, object], list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "sample_ids", "feature_source", "text_embeddings", "image_embeddings",
            "attribution_features", "model_family", "model_id", "model_revision",
            "tokenizer_revision", "preprocessing_sha256", "layer", "pooling",
            "id_column",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                "feature bundle lacks required dual-head provenance: "
                + ", ".join(sorted(missing))
            )
        sample_ids = [str(value).strip() for value in np.asarray(data["sample_ids"]).reshape(-1)]
        id_column = _npz_scalar_string(data, "id_column")
        pooling = _npz_scalar_string(data, "pooling")
        feature_source = _npz_scalar_string(data, "feature_source")
        text = np.asarray(data["text_embeddings"], dtype=np.float64)
        image = np.asarray(data["image_embeddings"], dtype=np.float64)
        attribution = np.asarray(data["attribution_features"], dtype=np.float64)
        bundle_provenance: dict[str, object] = {
            "model_family": _npz_scalar_string(data, "model_family"),
            "model_id": _npz_scalar_string(data, "model_id"),
            "model_revision": _npz_scalar_string(data, "model_revision"),
            "tokenizer_revision": _npz_scalar_string(data, "tokenizer_revision"),
            "preprocessing_sha256": _npz_scalar_string(data, "preprocessing_sha256"),
            "layer": _npz_scalar_int(data, "layer"),
        }

    if id_column != "variant_id":
        raise ValueError("bounded dual-head evasion requires features keyed by variant_id")
    if pooling != "text_image_tokens":
        raise ValueError(
            "dual-head evaluation requires a fused text_image_tokens feature bundle"
        )
    if any(not value for value in sample_ids):
        raise ValueError("feature bundle sample_ids must be non-empty")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("feature bundle sample_ids must be unique")
    expected_shape = (len(sample_ids), pair.base_feature_dim)
    if text.shape != expected_shape or image.shape != expected_shape:
        raise ValueError(
            "text and image primitive feature views must each have shape "
            f"{expected_shape}; got {text.shape} and {image.shape}"
        )
    if (
        attribution.ndim != 2
        or len(attribution) != len(sample_ids)
        or not np.all(np.isfinite(attribution))
        or not np.all(np.isfinite(text))
        or not np.all(np.isfinite(image))
    ):
        raise ValueError("feature arrays must be finite 2D arrays aligned by variant_id")

    shared = pair.manifest["shared_provenance"]
    mismatches: list[str] = []
    for key, observed in bundle_provenance.items():
        expected = shared[key]
        if key == "model_id":
            observed = str(observed).replace("\\", "/")
            expected = str(expected).replace("\\", "/")
        if observed != expected:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "feature provenance does not match both detector heads: "
            + ", ".join(mismatches)
        )
    result = {
        **bundle_provenance,
        "rows": len(sample_ids),
        "feature_source": feature_source,
        "id_column": id_column,
        "pooling": pooling,
        "text_feature_view": "text_representation",
        "text_pooling": "text_tokens",
        "text_embeddings_shape": list(text.shape),
        "image_feature_view": "image_representation",
        "image_pooling": "image_tokens",
        "image_embeddings_shape": list(image.shape),
        "attribution_features_shape": list(attribution.shape),
        "sample_ids_sha256": hashlib.sha256(
            json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    return result, sample_ids, np.concatenate([text, image], axis=1)


def _dual_decision(
    *, text_score: float, image_score: float, text_block: float,
    text_review: float, image_block: float, image_review: float,
    text_pooling: str, image_pooling: str,
) -> dict[str, object]:
    heads = {
        "text": _head_decision("text", text_score, text_block, text_review, text_pooling),
        "image": _head_decision("image", image_score, image_block, image_review, image_pooling),
    }
    winning_rank = max(int(row["action_rank"]) for row in heads.values())
    candidates = [
        row for row in heads.values() if int(row["action_rank"]) == winning_rank
    ]
    decisive = sorted(
        candidates,
        key=lambda row: (-float(row["band_progress"]), str(row["name"])),
    )[0]
    action = str(decisive["recommended_action"])
    return {
        "risk_score": decisive["risk_score"],
        "threshold": decisive["threshold"],
        "review_threshold": decisive["review_threshold"],
        "verdict": "malicious" if action == "block" else "benign",
        "uncertain": action == "review",
        "recommended_action": action,
        "decisive_head": decisive["name"],
        "decisive_band_progress": decisive["band_progress"],
        "head_decisions": heads,
    }


def _head_decision(
    name: str, score: float, block_threshold: float,
    review_threshold: float, pooling: str,
) -> dict[str, object]:
    values = (score, block_threshold, review_threshold)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} score and thresholds must be finite")
    if not 0.0 <= score <= 1.0 or not 0.0 <= review_threshold < block_threshold < 1.0:
        raise ValueError(f"{name} score or thresholds are outside their valid range")
    if score >= block_threshold:
        action = "block"
        progress = (score - block_threshold) / (1.0 - block_threshold)
    elif score >= review_threshold:
        action = "review"
        progress = (score - review_threshold) / (block_threshold - review_threshold)
    else:
        action = "allow"
        progress = score / review_threshold if review_threshold > 0.0 else 0.0
    return {
        "name": name,
        "risk_score": score,
        "threshold": block_threshold,
        "review_threshold": review_threshold,
        "recommended_action": action,
        "pooling": pooling,
        "band_progress": progress,
        "action_rank": ACTION_RANK[action],
    }


def _validate_dual_score_row(
    row: dict[str, str], decision: dict[str, object], manifest: dict[str, Any]
) -> None:
    if row["detector_mode"].strip() != "dual_or":
        raise ValueError("score-row detector_mode must be dual_or")
    expected_strings = {
        "recommended_action": decision["recommended_action"],
        "decisive_head": decision["decisive_head"],
        "verdict": decision["verdict"],
        "runtime_detector_identity_sha256": manifest["runtime_detector_identity_sha256"],
        "pair_identity_sha256": manifest["pair_identity_sha256"],
    }
    for key, expected in expected_strings.items():
        if row[key].strip() != expected:
            raise ValueError(f"score-row {key} does not match recomputed dual decision")
    if _parse_bool(row["uncertain"], "uncertain") is not decision["uncertain"]:
        raise ValueError("score-row uncertain does not match recomputed dual decision")

    _require_close_float(row["risk_score"], float(decision["risk_score"]), "risk_score")
    _require_exact_float(
        row["detector_threshold"], float(decision["threshold"]), "detector_threshold"
    )
    _require_exact_float(
        row["review_threshold"], float(decision["review_threshold"]), "review_threshold"
    )
    for name in ("text", "image"):
        expected = decision["head_decisions"][name]
        if row[f"{name}_action"].strip() != expected["recommended_action"]:
            raise ValueError(f"score-row {name}_action does not match recomputation")
        if row[f"{name}_pooling"].strip() != expected["pooling"]:
            raise ValueError(f"score-row {name}_pooling does not match its artifact")
        _require_close_float(
            row[f"{name}_risk_score"], float(expected["risk_score"]),
            f"{name}_risk_score",
        )
        _require_exact_float(
            row[f"{name}_block_threshold"], float(expected["threshold"]),
            f"{name}_block_threshold",
        )
        _require_exact_float(
            row[f"{name}_review_threshold"], float(expected["review_threshold"]),
            f"{name}_review_threshold",
        )


def _evaluate_legacy(
    args: argparse.Namespace, budgets: tuple[int, ...]
) -> dict[str, object]:
    detector = _detector_metadata(args.detector)
    artifact_threshold = float(detector["artifact_threshold"])
    threshold = float(args.threshold) if args.threshold is not None else artifact_threshold
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    if (
        args.threshold_kind == "detector_artifact_block_threshold"
        and threshold.hex() != artifact_threshold.hex()
    ):
        raise ValueError(
            "detector_artifact_block_threshold must equal the supplied detector artifact threshold"
        )

    feature_provenance, feature_ids = _legacy_feature_metadata(args.features)
    _validate_legacy_provenance(detector, feature_provenance)
    rows = _score_rows(args.scores, LEGACY_SCORE_FIELDS)
    _validate_panel_rows(rows, feature_ids)
    if any(not row["feature_view"].strip() for row in rows):
        raise ValueError("legacy feature_view values must be non-empty")
    recorded = [_probability(row["detector_threshold"], "detector_threshold") for row in rows]
    if any(value.hex() != artifact_threshold.hex() for value in recorded):
        raise ValueError("score-row detector threshold does not match the supplied artifact")

    evaluated: list[dict[str, object]] = []
    for row in rows:
        evaluated.append(
            {
                "base_sample_id": row["base_sample_id"].strip(),
                "query_index": int(row["query_index"]),
                "variant_type": row["variant_type"].strip(),
                "variant_id": row["variant_id"].strip(),
                "risk_score": _probability(row["risk_score"], "risk_score"),
            }
        )
    groups = _group_panel(evaluated)
    summary: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for budget in budgets:
        _require_budget_capacity(groups, budget)
        worst_scores: list[float] = []
        for sample_id, variants in sorted(groups.items()):
            eligible = sorted(variants, key=lambda row: int(row["query_index"]))[:budget]
            selected = min(
                eligible,
                key=lambda row: (float(row["risk_score"]), int(row["query_index"])),
            )
            score = float(selected["risk_score"])
            worst_scores.append(score)
            selections.append(
                {
                    "query_budget": budget,
                    "base_sample_id": sample_id,
                    "selected_variant": selected["variant_type"],
                    "selected_variant_id": selected["variant_id"],
                    "selected_query_index": selected["query_index"],
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

    return {
        "schema_version": 3,
        "method": "bounded_best_of_n_oracle_over_fixed_variant_order",
        "detector_mode": "legacy_single",
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
            "Legacy single-head detector-score oracle analysis. It is retained for "
            "historical comparisons and does not represent the v7 dual-head runtime. "
            "In shadow mode, threshold-based blocking is counterfactual because the "
            "effective runtime action remains allow."
        ),
        "summary": summary,
        "selections": selections,
    }


def _score_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    if not rows or required - fields:
        raise ValueError(f"scored variants must contain columns {sorted(required)}")
    return rows


def _validate_panel_rows(rows: list[dict[str, str]], feature_ids: list[str]) -> None:
    if any(row["label_id"].strip() != "1" for row in rows):
        raise ValueError("bounded evasion evaluation requires a malicious-only score panel")
    score_ids = [row["variant_id"].strip() for row in rows]
    if any(not value for value in score_ids):
        raise ValueError("scored variant_id values must be non-empty")
    if len(set(score_ids)) != len(score_ids):
        raise ValueError("scored variant_id values must be unique")
    if score_ids != feature_ids:
        raise ValueError("scored variant IDs are not aligned to the supplied feature bundle")
    if any(not row["base_sample_id"].strip() for row in rows):
        raise ValueError("base_sample_id values must be non-empty")
    if any(not row["variant_type"].strip() for row in rows):
        raise ValueError("variant_type values must be non-empty")


def _group_panel(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        try:
            query_index = int(row["query_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError("query_index values must be integers") from exc
        if query_index < 1:
            raise ValueError("query_index values must be positive")
        groups.setdefault(str(row["base_sample_id"]), []).append(row)
    for sample_id, variants in groups.items():
        indices = [int(row["query_index"]) for row in variants]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate query_index values for {sample_id}")
        if sorted(indices) != list(range(1, len(indices) + 1)):
            raise ValueError(
                f"query_index values must be contiguous from one for {sample_id}"
            )
    return groups


def _query_budgets(values: list[int]) -> tuple[int, ...]:
    budgets = tuple(sorted(set(values)))
    if not budgets or any(value < 1 for value in budgets):
        raise ValueError("query budgets must be positive")
    return budgets


def _require_budget_capacity(
    groups: dict[str, list[dict[str, object]]], budget: int
) -> None:
    underfilled = [
        sample_id for sample_id, variants in groups.items() if len(variants) < budget
    ]
    if underfilled:
        raise ValueError(f"query budget {budget} exceeds the available panel for {underfilled}")


def _detector_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        result: dict[str, object] = {
            "artifact_threshold": float(np.asarray(data["threshold"]).reshape(-1)[0]),
            "feature_dim": int(np.asarray(data["weights"]).size),
        }
        if "metadata_json" in data:
            result.update(json.loads(_npz_scalar_string(data, "metadata_json")))
        for key in (
            "source", "model_family", "model_id", "model_revision",
            "tokenizer_revision", "preprocessing_sha256", "pooling", "layer",
        ):
            if key in data:
                result[key] = (
                    _npz_scalar_int(data, key)
                    if key == "layer"
                    else _npz_scalar_string(data, key)
                )
    return result


def _legacy_feature_metadata(path: Path) -> tuple[dict[str, object], list[str]]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "sample_ids", "feature_source", "text_embeddings", "image_embeddings",
            "attribution_features", "model_family", "model_id", "model_revision",
            "tokenizer_revision", "preprocessing_sha256", "layer", "pooling",
            "id_column",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                "feature bundle lacks required provenance: " + ", ".join(sorted(missing))
            )
        sample_ids = [str(value).strip() for value in np.asarray(data["sample_ids"]).reshape(-1)]
        id_column = _npz_scalar_string(data, "id_column")
        result: dict[str, object] = {
            "rows": len(sample_ids),
            "feature_source": _npz_scalar_string(data, "feature_source"),
            "model_family": _npz_scalar_string(data, "model_family"),
            "model_id": _npz_scalar_string(data, "model_id"),
            "model_revision": _npz_scalar_string(data, "model_revision"),
            "tokenizer_revision": _npz_scalar_string(data, "tokenizer_revision"),
            "preprocessing_sha256": _npz_scalar_string(data, "preprocessing_sha256"),
            "layer": _npz_scalar_int(data, "layer"),
            "pooling": _npz_scalar_string(data, "pooling"),
            "id_column": id_column,
            "sample_ids_sha256": hashlib.sha256(
                json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
        for key in ("text_embeddings", "image_embeddings", "attribution_features"):
            result[f"{key}_shape"] = list(np.asarray(data[key]).shape)
    if id_column not in {"sample_id", "variant_id"}:
        raise ValueError("feature bundle id_column must be sample_id or variant_id")
    if any(not value for value in sample_ids):
        raise ValueError("feature bundle sample_ids must be non-empty")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("feature bundle sample_ids must be unique")
    if id_column != "variant_id":
        raise ValueError("bounded evasion evaluation requires features keyed by variant_id")
    return result, sample_ids


def _validate_legacy_provenance(
    detector: dict[str, object], features: dict[str, object]
) -> None:
    keys = (
        "model_family", "model_id", "model_revision", "tokenizer_revision",
        "preprocessing_sha256", "layer", "pooling",
    )
    missing = [key for key in keys if key not in detector]
    if missing:
        raise ValueError("detector lacks required provenance: " + ", ".join(missing))
    mismatches: list[str] = []
    for key in keys:
        detector_value = detector[key]
        feature_value = features[key]
        if key == "model_id":
            detector_value = str(detector_value).replace("\\", "/")
            feature_value = str(feature_value).replace("\\", "/")
        if detector_value != feature_value:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "feature provenance does not match detector: " + ", ".join(mismatches)
        )


def _npz_scalar_string(data: np.lib.npyio.NpzFile, name: str) -> str:
    values = np.asarray(data[name]).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"{name} must contain one scalar value")
    value = str(values[0]).strip()
    if not value:
        raise ValueError(f"{name} provenance must be non-empty")
    return value


def _npz_scalar_int(data: np.lib.npyio.NpzFile, name: str) -> int:
    values = np.asarray(data[name]).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"{name} must contain one scalar value")
    value = values[0]
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"score-row {name} must be true or false")


def _probability(value: str, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"score-row {name} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"score-row {name} must be finite and in [0,1]")
    return result


def _require_close_float(value: str, expected: float, name: str) -> None:
    observed = _probability(value, name)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"score-row {name} does not match artifact recomputation")


def _require_exact_float(value: str, expected: float, name: str) -> None:
    observed = _probability(value, name)
    if observed.hex() != float(expected).hex():
        raise ValueError(f"score-row {name} does not exactly match its bound threshold")


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
