from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DELIVERABLES = ROOT.parent
REPOSITORY = DELIVERABLES.parent
PIPELINE = DELIVERABLES / "pipeline"
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(PIPELINE))

from AEGIS.deployment_config import (  # noqa: E402
    DeploymentConfig,
    load_configured_detector,
    load_deployment_config,
)
from AEGIS.detector_artifact import DetectorArtifact  # noqa: E402
from AEGIS.detector_set import OrDetector  # noqa: E402
from aegis_research.bordair_dual import (  # noqa: E402
    load_detector_pair,
    score_pair,
)
from aegis_research.experiment import ExperimentConfig, fit_evaluate_view, run_ablation_suite  # noqa: E402
from aegis_research.io import load_feature_bundle, load_metadata  # noqa: E402
from aegis_research.signals import (  # noqa: E402
    POOLING_FEATURE_VIEWS,
    binary_entropy,
    build_feature_views,
)


DEFAULT_METADATA = DELIVERABLES / "benchmark" / "data" / "benchmark.csv"
PROVENANCE_KEYS = (
    "model_family",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "preprocessing_sha256",
    "layer",
    "pooling",
)
COMMON_PROVENANCE_KEYS = tuple(
    key for key in PROVENANCE_KEYS if key != "pooling"
)
ACTION_RANK = {"allow": 0, "review": 1, "block": 2}
RUNTIME_CONFIGS = {
    "llava05b": "aegis.llava.deployment.container.json",
    "qwen25vl3b": "aegis.deployment.container.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the AEGIS ablation and transfer report.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Provenance-complete MLLM feature bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "current",
        help="Run-specific output directory.",
    )
    parser.add_argument("--report", type=Path, help="Report path; defaults to OUTPUT_DIR/REPORT.md.")
    parser.add_argument("--adaptive-summary", type=Path)
    parser.add_argument("--adaptive-scores", type=Path)
    parser.add_argument("--adaptive-detector", type=Path)
    parser.add_argument(
        "--adaptive-pair-manifest",
        type=Path,
        help=(
            "Canonical v7 detector-pair manifest used to produce dual-head "
            "adaptive scores. --adaptive-detector is the legacy single-head option."
        ),
    )
    parser.add_argument("--adaptive-features", type=Path)
    parser.add_argument(
        "--runtime-target",
        choices=("llava05b", "qwen25vl3b"),
        default="llava05b",
        help="Built-in target used for the runtime relationship and compatible scoring.",
    )
    parser.add_argument("--force", action="store_true", help="Allow replacement of an existing run directory.")
    args = parser.parse_args()
    orphaned_adaptive = [
        name
        for name, value in (
            ("--adaptive-scores", args.adaptive_scores),
            ("--adaptive-detector", args.adaptive_detector),
            ("--adaptive-pair-manifest", args.adaptive_pair_manifest),
            ("--adaptive-features", args.adaptive_features),
        )
        if value is not None and args.adaptive_summary is None
    ]
    if orphaned_adaptive:
        parser.error(
            "--adaptive-summary is required when using "
            + ", ".join(orphaned_adaptive)
        )
    reproduction_lines = _reproduction_lines(
        runtime_target=args.runtime_target,
        metadata=args.metadata,
        features=args.features,
        output_dir=args.output_dir,
    )
    report_path = args.report or args.output_dir / "REPORT.md"
    _prepare_output(args.output_dir, report_path, force=args.force)
    feature_provenance = _load_feature_provenance(args.features)
    if feature_provenance["id_column"] != "sample_id":
        raise ValueError("ablation requires a feature bundle keyed by sample_id")
    metadata = load_metadata(args.metadata)
    bundle = load_feature_bundle(args.features, metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runtime_context, runtime_detector, runtime_config = _current_runtime(
        args.runtime_target
    )
    runtime_rows, runtime_summary, runtime_scoring = _score_runtime_detector(
        metadata=metadata,
        bundle=bundle,
        feature_provenance=feature_provenance,
        detector=runtime_detector,
        config=runtime_config,
    )
    if runtime_rows:
        runtime_scores_path = args.output_dir / "runtime_detector_scores.csv"
        runtime_summary_path = args.output_dir / "runtime_detector_summary.csv"
        _write_csv(runtime_scores_path, runtime_rows)
        _write_csv(runtime_summary_path, runtime_summary)
        runtime_scoring.update(
            {
                "results": runtime_summary,
                "scores": _portable_path(runtime_scores_path),
                "scores_sha256": _sha256(runtime_scores_path),
                "summary": _portable_path(runtime_summary_path),
                "summary_sha256": _sha256(runtime_summary_path),
            }
        )

    full_results, fitted = run_ablation_suite(metadata, bundle, ExperimentConfig(seed=42))
    all_metrics = fitted["all_input_signals"]
    full_rows = _with_importance(full_results)
    _write_csv(args.output_dir / "feature_signal_ablation.csv", full_rows)

    low_label_runs: list[dict[str, object]] = []
    for trusted in (4, 8):
        for pseudo in (0, 4):
            for seed in (11, 22, 33, 44, 55):
                config = ExperimentConfig(seed=seed, trusted_per_class=trusted, pseudo_per_class=pseudo)
                results, _ = run_ablation_suite(metadata, bundle, config)
                selected = next(row for row in results if row["feature_view"] == "all_input_signals")
                low_label_runs.append({"trusted_per_class": trusted, "pseudo_per_class": pseudo, "seed": seed, **selected})
    _write_csv(args.output_dir / "low_label_seed_runs.csv", low_label_runs)
    low_label_summary = _aggregate_low_label(low_label_runs)
    _write_csv(args.output_dir / "low_label_summary.csv", low_label_summary)

    transfer_rows = _attack_family_transfer(metadata, bundle)
    _write_csv(args.output_dir / "attack_family_transfer.csv", transfer_rows)
    uncertainty_rows = _uncertainty_coverage(metadata, all_metrics)
    _write_csv(args.output_dir / "uncertainty_coverage.csv", uncertainty_rows)
    adaptive_rows: list[dict[str, object]] = []
    adaptive_provenance = None
    if args.adaptive_summary is not None:
        adaptive_rows, adaptive_provenance = _load_adaptive_evidence(args, parser)
        _write_csv(args.output_dir / "adaptive_redteam_summary.csv", adaptive_rows)

    provenance = {
        "metadata": _portable_path(args.metadata),
        "metadata_sha256": _sha256(args.metadata),
        "features": _portable_path(args.features),
        "features_sha256": _sha256(args.features),
        "feature_source": bundle.feature_source,
        "feature_provenance": feature_provenance,
        "rows": len(metadata),
        "selection": "feature view and threshold use validation data only",
        "test_policy": "test rows are not used for hyperparameter or threshold selection",
        "scientific_status": "provenance_complete_model_evidence",
        "adaptive_evidence": adaptive_provenance,
        "runtime_context": runtime_context,
        "runtime_scoring": runtime_scoring,
        "reproduction": {
            "runtime_target": args.runtime_target,
            "run_root": _portable_path(args.output_dir.resolve().parent),
            "metadata": _portable_path(args.metadata),
            "features": _portable_path(args.features),
            "output_dir": _portable_path(args.output_dir),
            "command": reproduction_lines,
        },
        "tool_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = _render_report(full_rows, low_label_summary, transfer_rows, uncertainty_rows, adaptive_rows, provenance)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": _portable_path(report_path), **provenance}, indent=2))


def _prepare_output(output_dir: Path, report_path: Path, *, force: bool) -> None:
    existing = []
    if output_dir.exists():
        existing.extend(path for path in output_dir.iterdir())
    if report_path.exists() and report_path.parent != output_dir:
        existing.append(report_path)
    if existing and not force:
        raise FileExistsError(
            f"refusing to overwrite an existing evidence run at {output_dir}; "
            "choose a new --output-dir or pass --force explicitly"
        )


def _load_adaptive_evidence(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    required = {
        "--adaptive-scores": args.adaptive_scores,
        "--adaptive-features": args.adaptive_features,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(
            "--adaptive-summary requires " + ", ".join(sorted(missing))
        )
    if (args.adaptive_detector is None) == (args.adaptive_pair_manifest is None):
        parser.error(
            "--adaptive-summary requires exactly one of --adaptive-pair-manifest "
            "(v7 dual-head) or --adaptive-detector (legacy single-head)"
        )
    if args.adaptive_pair_manifest is not None:
        return _load_dual_adaptive_evidence(args)
    return _load_legacy_adaptive_evidence(args)


def _load_legacy_adaptive_evidence(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    assert args.adaptive_detector is not None
    assert args.adaptive_scores is not None
    assert args.adaptive_features is not None
    payload = json.loads(args.adaptive_summary.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 3
        or payload.get("detector_mode") != "legacy_single"
    ):
        raise ValueError(
            "legacy adaptive summary must use schema 3 and detector_mode legacy_single"
        )
    if payload.get("method") != "bounded_best_of_n_oracle_over_fixed_variant_order":
        raise ValueError("legacy adaptive summary method is unsupported")
    rows = payload.get("summary")
    if not isinstance(rows, list) or not rows or any(
        not isinstance(row, dict) for row in rows
    ):
        raise ValueError("adaptive summary does not contain result rows")
    selections = payload.get("selections")
    if not isinstance(selections, list) or not selections or any(
        not isinstance(row, dict) for row in selections
    ):
        raise ValueError("legacy adaptive summary does not contain oracle selections")
    summary_provenance = payload.get("provenance")
    if not isinstance(summary_provenance, dict):
        raise ValueError("adaptive summary does not contain provenance")
    supplied_hashes = {
        "scores_sha256": _sha256(args.adaptive_scores),
        "detector_sha256": _sha256(args.adaptive_detector),
        "features_sha256": _sha256(args.adaptive_features),
    }
    hash_mismatches = [
        key
        for key, actual in supplied_hashes.items()
        if summary_provenance.get(key) != actual
    ]
    if hash_mismatches:
        raise ValueError(
            "adaptive summary does not describe the supplied inputs: "
            + ", ".join(sorted(hash_mismatches))
        )
    expected_paths = {
        "scores": _portable_path(args.adaptive_scores),
        "detector": _portable_path(args.adaptive_detector),
        "features": _portable_path(args.adaptive_features),
    }
    for key, expected in expected_paths.items():
        if summary_provenance.get(key) != expected:
            raise ValueError(f"legacy adaptive provenance path differs for {key}")

    detector_context = _load_detector_context(args.adaptive_detector)
    detector_metadata = _load_legacy_detector_metadata(args.adaptive_detector)
    if summary_provenance.get("detector_metadata") != detector_metadata:
        raise ValueError("legacy adaptive detector metadata is not exact")
    detector_threshold = float(detector_context["threshold"])
    adaptive_feature_provenance, feature_ids, selected_features = (
        _legacy_feature_snapshot(
            args.adaptive_features,
            detector_context,
        )
    )
    if summary_provenance.get("feature_provenance") != adaptive_feature_provenance:
        raise ValueError("legacy adaptive feature provenance is not exact")
    threshold_kind = summary_provenance.get("threshold_kind")
    if threshold_kind != "detector_artifact_block_threshold":
        raise ValueError(
            "ablation import requires detector_artifact_block_threshold evidence"
        )
    for key in ("threshold", "detector_artifact_threshold"):
        value = summary_provenance.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value).hex() != detector_threshold.hex()
        ):
            raise ValueError(f"legacy adaptive {key} differs from the detector")
    if summary_provenance.get("traffic_mode") not in {
        "not_applicable",
        "shadow",
        "review",
        "enforce",
    }:
        raise ValueError("legacy adaptive traffic mode is invalid")

    with args.adaptive_scores.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        score_rows = list(reader)
        score_fields = set(reader.fieldnames or ())
    required_score_fields = {
        "base_sample_id",
        "query_index",
        "variant_type",
        "variant_id",
        "label_id",
        "detector_mode",
        "risk_score",
        "detector_threshold",
        "feature_view",
    }
    if not score_rows or required_score_fields - score_fields:
        raise ValueError("legacy adaptive scores are missing required fields")
    score_ids = [row["variant_id"].strip() for row in score_rows]
    if score_ids != feature_ids:
        raise ValueError(
            "legacy adaptive score variant IDs are not exactly aligned to features"
        )
    if any(row["label_id"].strip() != "1" for row in score_rows):
        raise ValueError("legacy adaptive scores must be a malicious-only panel")
    if any(row["detector_mode"].strip() != "legacy_single" for row in score_rows):
        raise ValueError("legacy adaptive score row has the wrong detector mode")
    if any(
        row["feature_view"].strip() != detector_context["feature_view"]
        for row in score_rows
    ):
        raise ValueError("legacy adaptive score feature view is incompatible")

    recomputed_scores = _score_legacy_features(
        args.adaptive_detector,
        selected_features,
    )
    recomputed_rows: list[dict[str, object]] = []
    for index, (row, recomputed_score) in enumerate(
        zip(score_rows, recomputed_scores)
    ):
        if float(row["detector_threshold"]).hex() != detector_threshold.hex():
            raise ValueError(
                f"legacy adaptive score row {index} threshold is not exact"
            )
        observed_score = float(row["risk_score"])
        if (
            not math.isfinite(observed_score)
            or not 0.0 <= observed_score <= 1.0
            or not math.isclose(
                observed_score,
                float(recomputed_score),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"legacy adaptive score row {index} differs from recomputation"
            )
        base_sample_id = row["base_sample_id"].strip()
        variant_type = row["variant_type"].strip()
        if not base_sample_id or not variant_type:
            raise ValueError("legacy adaptive panel identifiers must be non-empty")
        try:
            query_index = int(row["query_index"])
        except ValueError as exc:
            raise ValueError("legacy adaptive query_index must be an integer") from exc
        recomputed_rows.append(
            {
                "base_sample_id": base_sample_id,
                "query_index": query_index,
                "variant_type": variant_type,
                "variant_id": row["variant_id"].strip(),
                "risk_score": float(recomputed_score),
            }
        )

    budgets = [int(row.get("query_budget", 0)) for row in rows]
    if budgets != sorted(set(budgets)) or any(value < 1 for value in budgets):
        raise ValueError("legacy adaptive query budgets are not canonical")
    expected_rows, expected_selections = _recompute_legacy_best_of_n(
        recomputed_rows,
        budgets,
        detector_threshold,
    )
    _require_evidence_rows_equal(rows, expected_rows, "legacy adaptive summary")
    _require_evidence_rows_equal(
        selections,
        expected_selections,
        "legacy adaptive selections",
    )
    return list(rows), {
        "detector_mode": "legacy_single",
        "compatibility_note": (
            "Legacy single-head adaptive evidence; not a v7 dual-head OR pair."
        ),
        "summary": _portable_path(args.adaptive_summary),
        "summary_sha256": _sha256(args.adaptive_summary),
        "scores": _portable_path(args.adaptive_scores),
        "scores_sha256": supplied_hashes["scores_sha256"],
        "detector": _portable_path(args.adaptive_detector),
        "detector_sha256": supplied_hashes["detector_sha256"],
        "features": _portable_path(args.adaptive_features),
        "features_sha256": supplied_hashes["features_sha256"],
        "feature_provenance": adaptive_feature_provenance,
        "feature_view": str(detector_context["feature_view"]),
        "threshold": detector_threshold,
        "threshold_kind": threshold_kind,
        "selection_policy": (
            "lowest score within the first N fixed query-index variants, then "
            "earliest query index"
        ),
    }


def _load_legacy_detector_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        if "threshold" not in data or "weights" not in data:
            raise ValueError("legacy adaptive detector lacks threshold or weights")
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
            "layer",
        ):
            if key in data:
                value = np.asarray(data[key]).reshape(-1)[0]
                result[key] = int(value) if key == "layer" else str(value)
    return result


def _legacy_feature_snapshot(
    path: Path,
    detector_context: dict[str, object],
) -> tuple[dict[str, object], list[str], np.ndarray]:
    raw = _load_feature_provenance(path)
    if raw["id_column"] != "variant_id":
        raise ValueError("legacy adaptive features must be keyed by variant_id")
    _validate_detector_feature_provenance(detector_context, raw)
    detector_source = detector_context.get("feature_source")
    if detector_source is not None and detector_source != raw["feature_source"]:
        raise ValueError("legacy adaptive feature_source differs from detector")
    with np.load(path, allow_pickle=False) as data:
        sample_ids = [
            str(value).strip()
            for value in np.asarray(data["sample_ids"]).reshape(-1)
        ]
        text = np.asarray(data["text_embeddings"], dtype=np.float64)
        image = np.asarray(data["image_embeddings"], dtype=np.float64)
        attribution = np.asarray(data["attribution_features"], dtype=np.float64)
    views = build_feature_views(text, image, attribution)
    view_name = str(detector_context["feature_view"])
    if view_name not in views:
        raise ValueError("legacy adaptive detector feature view is unsupported")
    selected = np.asarray(views[view_name], dtype=np.float64)
    if selected.shape != (len(sample_ids), int(detector_context["feature_dim"])):
        raise ValueError("legacy adaptive feature dimension differs from detector")
    provenance = {
        "rows": raw["rows"],
        "feature_source": raw["feature_source"],
        **{key: raw[key] for key in PROVENANCE_KEYS},
        "id_column": raw["id_column"],
        "sample_ids_sha256": raw["sample_ids_sha256"],
        "text_embeddings_shape": [raw["rows"], raw["text_feature_dim"]],
        "image_embeddings_shape": [raw["rows"], raw["image_feature_dim"]],
        "attribution_features_shape": [
            raw["rows"],
            raw["attribution_feature_dim"],
        ],
    }
    return provenance, sample_ids, selected


def _score_legacy_features(path: Path, features: np.ndarray) -> np.ndarray:
    selected = np.asarray(features, dtype=np.float64)
    with np.load(path, allow_pickle=False) as data:
        required = {"weights", "mean", "scale", "bias", "threshold"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                "legacy adaptive detector lacks scoring arrays: "
                + ", ".join(sorted(missing))
            )
        weights = np.asarray(data["weights"], dtype=np.float64).reshape(-1)
        mean = np.asarray(data["mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(data["scale"], dtype=np.float64).reshape(-1)
        bias = float(np.asarray(data["bias"]).reshape(-1)[0])
    if (
        selected.ndim != 2
        or weights.shape != mean.shape
        or weights.shape != scale.shape
        or weights.shape != (selected.shape[1],)
    ):
        raise ValueError("legacy adaptive detector scoring dimensions are incompatible")
    if (
        not np.all(np.isfinite(selected))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(scale))
        or not math.isfinite(bias)
        or np.any(scale <= 0.0)
    ):
        raise ValueError("legacy adaptive detector scoring values are invalid")
    logits = ((selected - mean) / scale) @ weights + bias
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _recompute_legacy_best_of_n(
    rows: list[dict[str, object]],
    budgets: list[int],
    threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        query_index = int(row["query_index"])
        if query_index < 1:
            raise ValueError("legacy adaptive query indices must be positive")
        groups.setdefault(str(row["base_sample_id"]), []).append(row)
    if not groups:
        raise ValueError("legacy adaptive panel is empty")
    for sample_id, variants in groups.items():
        indices = [int(row["query_index"]) for row in variants]
        if sorted(indices) != list(range(1, len(indices) + 1)):
            raise ValueError(
                f"legacy adaptive query indices are not contiguous for {sample_id}"
            )

    summaries: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for budget in budgets:
        underfilled = [
            sample_id
            for sample_id, variants in groups.items()
            if len(variants) < budget
        ]
        if underfilled:
            raise ValueError(
                f"legacy adaptive query budget {budget} exceeds {underfilled}"
            )
        worst_scores: list[float] = []
        for sample_id, variants in sorted(groups.items()):
            eligible = sorted(
                variants,
                key=lambda row: int(row["query_index"]),
            )[:budget]
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
                    "selected_variant_id": selected["variant_id"],
                    "selected_query_index": selected["query_index"],
                    "selected_risk_score": score,
                    "evaded": score < threshold,
                }
            )
        detected = sum(score >= threshold for score in worst_scores)
        summaries.append(
            {
                "query_budget": budget,
                "n_samples": len(worst_scores),
                "threshold": threshold,
                "malicious_recall": detected / len(worst_scores),
                "evasion_rate": 1.0 - detected / len(worst_scores),
                "mean_worst_case_score": sum(worst_scores) / len(worst_scores),
            }
        )
    return summaries, selections


def _require_evidence_rows_equal(
    observed: list[dict[str, object]],
    expected: list[dict[str, object]],
    context: str,
) -> None:
    if len(observed) != len(expected):
        raise ValueError(f"{context} row count differs from recomputation")
    for index, (actual_row, expected_row) in enumerate(zip(observed, expected)):
        if set(actual_row) != set(expected_row):
            raise ValueError(f"{context} row {index} fields are not canonical")
        for key, expected_value in expected_row.items():
            actual_value = actual_row[key]
            if isinstance(expected_value, bool):
                matches = isinstance(actual_value, bool) and actual_value is expected_value
            elif isinstance(expected_value, int):
                matches = (
                    isinstance(actual_value, int)
                    and not isinstance(actual_value, bool)
                    and actual_value == expected_value
                )
            elif isinstance(expected_value, float):
                matches = (
                    isinstance(actual_value, (int, float))
                    and not isinstance(actual_value, bool)
                    and math.isclose(
                        float(actual_value),
                        expected_value,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )
            else:
                matches = actual_value == expected_value
            if not matches:
                raise ValueError(
                    f"{context} row {index} differs for {key}"
                )


def _load_dual_adaptive_evidence(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    assert args.adaptive_pair_manifest is not None
    assert args.adaptive_scores is not None
    assert args.adaptive_features is not None
    payload = json.loads(args.adaptive_summary.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 3 or payload.get("detector_mode") != "dual_or":
        raise ValueError(
            "v7 dual-head adaptive summary must use schema 3 and detector_mode dual_or"
        )
    if payload.get("method") != "bounded_best_of_n_oracle_over_fixed_variant_order":
        raise ValueError("dual adaptive summary method is unsupported")
    rows = payload.get("summary")
    if not isinstance(rows, list) or not rows or any(
        not isinstance(row, dict) for row in rows
    ):
        raise ValueError("adaptive summary does not contain result rows")
    summary_provenance = payload.get("provenance")
    if not isinstance(summary_provenance, dict):
        raise ValueError("adaptive summary does not contain provenance")

    supplied_hashes = {
        "scores_sha256": _sha256(args.adaptive_scores),
        "features_sha256": _sha256(args.adaptive_features),
        "pair_manifest_sha256": _sha256(args.adaptive_pair_manifest),
    }
    hash_mismatches = [
        key
        for key, actual in supplied_hashes.items()
        if summary_provenance.get(key) != actual
    ]
    if hash_mismatches:
        raise ValueError(
            "adaptive summary does not describe the supplied inputs: "
            + ", ".join(sorted(hash_mismatches))
        )

    pair = load_detector_pair(args.adaptive_pair_manifest)
    manifest = pair.manifest
    for key in ("pair_identity_sha256", "runtime_detector_identity_sha256"):
        if summary_provenance.get(key) != manifest.get(key):
            raise ValueError(f"adaptive summary {key} does not match the pair manifest")
    if summary_provenance.get("target") != manifest.get("target"):
        raise ValueError("adaptive summary target does not match the pair manifest")
    if summary_provenance.get("composition") != "or":
        raise ValueError("dual adaptive summary must record OR composition")
    expected_oracle_order = (
        "lowest_aggregate_action_rank_then_lowest_normalized_"
        "decisive_band_progress_then_earliest_query_index"
    )
    if summary_provenance.get("oracle_order") != expected_oracle_order:
        raise ValueError("dual adaptive summary oracle ordering is invalid")
    reported_artifacts = summary_provenance.get("artifacts")
    if not isinstance(reported_artifacts, dict) or set(reported_artifacts) != {
        "text",
        "image",
    }:
        raise ValueError("dual adaptive summary must bind text and image artifacts")
    for name in ("text", "image"):
        reported = reported_artifacts[name]
        expected = manifest["artifacts"][name]
        if not isinstance(reported, dict):
            raise ValueError(f"adaptive summary {name} artifact binding is invalid")
        for key in (
            "sha256",
            "pooling",
            "feature_dim",
            "block_threshold",
            "review_threshold",
            "source",
        ):
            if reported.get(key) != expected.get(key):
                raise ValueError(
                    f"adaptive summary {name} artifact differs for {key}"
                )

    adaptive_feature_provenance, feature_ids, fused = _dual_feature_snapshot(
        args.adaptive_features,
        pair,
    )
    reported_features = summary_provenance.get("feature_provenance")
    if not isinstance(reported_features, dict):
        raise ValueError("adaptive summary does not bind feature provenance")
    for key, value in adaptive_feature_provenance.items():
        if reported_features.get(key) != value:
            raise ValueError(
                f"adaptive summary feature provenance differs for {key}"
            )

    with args.adaptive_scores.open("r", encoding="utf-8-sig", newline="") as handle:
        score_rows = list(csv.DictReader(handle))
    required_score_fields = {
        "base_sample_id",
        "query_index",
        "variant_type",
        "variant_id",
        "label_id",
        "detector_mode",
        "risk_score",
        "detector_threshold",
        "review_threshold",
        "verdict",
        "uncertain",
        "recommended_action",
        "decisive_head",
        "text_risk_score",
        "text_block_threshold",
        "text_review_threshold",
        "text_action",
        "text_pooling",
        "image_risk_score",
        "image_block_threshold",
        "image_review_threshold",
        "image_action",
        "image_pooling",
        "runtime_detector_identity_sha256",
        "pair_identity_sha256",
    }
    if not score_rows or required_score_fields - set(score_rows[0]):
        raise ValueError(
            "dual adaptive scores are missing required per-head decision fields"
        )
    score_ids = [row["variant_id"].strip() for row in score_rows]
    if score_ids != feature_ids:
        raise ValueError(
            "adaptive score variant IDs are not exactly aligned to the feature bundle"
        )
    if any(row["label_id"].strip() != "1" for row in score_rows):
        raise ValueError("adaptive scores must be a malicious-only panel")

    recomputed = score_pair(pair, fused)
    entries = manifest["artifacts"]
    recomputed_rows: list[dict[str, object]] = []
    for index, row in enumerate(score_rows):
        if row["detector_mode"] != "dual_or":
            raise ValueError("adaptive score row is not dual_or")
        head_rows: dict[str, dict[str, object]] = {}
        for name in ("text", "image"):
            score = float(recomputed[f"{name}_scores"][index])
            action = str(recomputed[f"{name}_actions"][index])
            entry = entries[name]
            expected_values = {
                f"{name}_risk_score": (score, False),
                f"{name}_block_threshold": (
                    float(entry["block_threshold"]),
                    True,
                ),
                f"{name}_review_threshold": (
                    float(entry["review_threshold"]),
                    True,
                ),
            }
            for field, (expected, exact) in expected_values.items():
                observed = float(row[field])
                if (
                    observed != expected
                    if exact
                    else not math.isclose(
                        observed, expected, rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    raise ValueError(
                        f"adaptive score row {index} differs from {name} artifact for {field}"
                    )
            if row[f"{name}_action"] != action:
                raise ValueError(
                    f"adaptive score row {index} has an invalid {name} action"
                )
            if row[f"{name}_pooling"] != entry["pooling"]:
                raise ValueError(
                    f"adaptive score row {index} has invalid {name} pooling"
                )
            head_rows[name] = {
                "score": score,
                "action": action,
                "block": float(entry["block_threshold"]),
                "review": float(entry["review_threshold"]),
            }
        decisive = _decisive_head(head_rows)
        action = str(head_rows[decisive]["action"])
        exact_strings = {
            "recommended_action": action,
            "decisive_head": decisive,
            "verdict": "malicious" if action == "block" else "benign",
            "uncertain": str(action == "review"),
            "runtime_detector_identity_sha256": manifest[
                "runtime_detector_identity_sha256"
            ],
            "pair_identity_sha256": manifest["pair_identity_sha256"],
        }
        for field, expected in exact_strings.items():
            if row[field].strip().lower() != str(expected).lower():
                raise ValueError(
                    f"adaptive score row {index} has invalid {field}"
                )
        decisive_values = {
            "risk_score": (float(head_rows[decisive]["score"]), False),
            "detector_threshold": (float(head_rows[decisive]["block"]), True),
            "review_threshold": (float(head_rows[decisive]["review"]), True),
        }
        for field, (expected, exact) in decisive_values.items():
            observed = float(row[field])
            if (
                observed != expected
                if exact
                else not math.isclose(
                    observed, expected, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise ValueError(
                    f"adaptive score row {index} has invalid decisive {field}"
                )
        recomputed_rows.append(
            {
                "base_sample_id": row["base_sample_id"].strip(),
                "query_index": int(row["query_index"]),
                "variant_id": row["variant_id"].strip(),
                "recommended_action": action,
                "decisive_head": decisive,
                "decisive_band_progress": _head_progress(
                    head_rows[decisive]
                ),
                "text_action": head_rows["text"]["action"],
                "image_action": head_rows["image"]["action"],
            }
        )

    required_summary_fields = {
        "query_budget",
        "n_samples",
        "malicious_recall",
        "evasion_rate",
        "review_or_block_recall",
        "allow_rate",
        "review_rate",
        "block_rate",
        "per_head_block_recall",
        "decisive_head_counts",
        "mean_decisive_band_progress",
    }
    if any(required_summary_fields - set(row) for row in rows):
        raise ValueError("dual adaptive summary rows are incomplete")
    _validate_dual_adaptive_summary(list(rows), recomputed_rows)
    return list(rows), {
        "detector_mode": "dual_or",
        "summary": _portable_path(args.adaptive_summary),
        "summary_sha256": _sha256(args.adaptive_summary),
        "scores": _portable_path(args.adaptive_scores),
        "scores_sha256": supplied_hashes["scores_sha256"],
        "pair_manifest": _portable_path(args.adaptive_pair_manifest),
        "pair_manifest_sha256": supplied_hashes["pair_manifest_sha256"],
        "pair_identity_sha256": manifest["pair_identity_sha256"],
        "runtime_detector_identity_sha256": manifest[
            "runtime_detector_identity_sha256"
        ],
        "artifacts": reported_artifacts,
        "features": _portable_path(args.adaptive_features),
        "features_sha256": supplied_hashes["features_sha256"],
        "feature_provenance": adaptive_feature_provenance,
        "composition": "Block > Review > Allow",
        "selection_policy": expected_oracle_order,
    }


def _dual_feature_snapshot(
    path: Path,
    pair: Any,
) -> tuple[dict[str, object], list[str], np.ndarray]:
    raw = _load_feature_provenance(path)
    if raw["id_column"] != "variant_id":
        raise ValueError("dual adaptive features must be keyed by variant_id")
    if raw["pooling"] != "text_image_tokens":
        raise ValueError("dual adaptive features require text_image_tokens pooling")
    shared = pair.manifest["shared_provenance"]
    mismatches: list[str] = []
    for key in COMMON_PROVENANCE_KEYS:
        observed = raw[key]
        expected = shared[key]
        if key == "model_id":
            observed = _normalized_model_id(observed)
            expected = _normalized_model_id(expected)
        if observed != expected:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "dual adaptive feature provenance does not match both heads: "
            + ", ".join(mismatches)
        )
    if (
        raw["text_feature_dim"] != pair.base_feature_dim
        or raw["image_feature_dim"] != pair.base_feature_dim
    ):
        raise ValueError("dual adaptive primitive feature dimensions are incompatible")
    with np.load(path, allow_pickle=False) as data:
        sample_ids = [
            str(value).strip()
            for value in np.asarray(data["sample_ids"]).reshape(-1)
        ]
        text = np.asarray(data["text_embeddings"], dtype=np.float64)
        image = np.asarray(data["image_embeddings"], dtype=np.float64)
    provenance = {
        **{key: raw[key] for key in COMMON_PROVENANCE_KEYS},
        "rows": raw["rows"],
        "feature_source": raw["feature_source"],
        "id_column": raw["id_column"],
        "pooling": raw["pooling"],
        "text_feature_view": "text_representation",
        "text_pooling": "text_tokens",
        "text_embeddings_shape": [raw["rows"], raw["text_feature_dim"]],
        "image_feature_view": "image_representation",
        "image_pooling": "image_tokens",
        "image_embeddings_shape": [raw["rows"], raw["image_feature_dim"]],
        "attribution_features_shape": [
            raw["rows"],
            raw["attribution_feature_dim"],
        ],
        "sample_ids_sha256": raw["sample_ids_sha256"],
    }
    return provenance, sample_ids, np.concatenate([text, image], axis=1)


def _validate_dual_adaptive_summary(
    summaries: list[dict[str, object]],
    score_rows: list[dict[str, object]],
) -> None:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in score_rows:
        sample_id = str(row["base_sample_id"])
        if not sample_id:
            raise ValueError("dual adaptive base_sample_id must be non-empty")
        groups.setdefault(sample_id, []).append(row)
    if not groups:
        raise ValueError("dual adaptive panel is empty")
    for sample_id, variants in groups.items():
        indices = [int(row["query_index"]) for row in variants]
        if sorted(indices) != list(range(1, len(indices) + 1)):
            raise ValueError(
                f"dual adaptive query indices are not contiguous for {sample_id}"
            )

    budgets = [int(row["query_budget"]) for row in summaries]
    if any(value < 1 for value in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("dual adaptive query budgets must be unique and positive")
    for summary, budget in zip(summaries, budgets):
        if any(len(variants) < budget for variants in groups.values()):
            raise ValueError("dual adaptive query budget exceeds its fixed panel")
        selected = [
            min(
                sorted(variants, key=lambda row: int(row["query_index"]))[:budget],
                key=lambda row: (
                    ACTION_RANK[str(row["recommended_action"])],
                    float(row["decisive_band_progress"]),
                    int(row["query_index"]),
                ),
            )
            for _, variants in sorted(groups.items())
        ]
        count = len(selected)
        actions = [str(row["recommended_action"]) for row in selected]
        action_counts = {
            action: actions.count(action) for action in ("allow", "review", "block")
        }
        head_blocks = {
            name: sum(row[f"{name}_action"] == "block" for row in selected)
            for name in ("text", "image")
        }
        decisive_counts = {
            name: sum(row["decisive_head"] == name for row in selected)
            for name in ("text", "image")
        }
        expected_scalars = {
            "n_samples": count,
            "malicious_recall": action_counts["block"] / count,
            "evasion_rate": 1.0 - action_counts["block"] / count,
            "review_or_block_recall": (
                action_counts["review"] + action_counts["block"]
            )
            / count,
            "allow_rate": action_counts["allow"] / count,
            "review_rate": action_counts["review"] / count,
            "block_rate": action_counts["block"] / count,
            "mean_decisive_band_progress": float(
                np.mean(
                    [float(row["decisive_band_progress"]) for row in selected]
                )
            ),
        }
        for field, expected in expected_scalars.items():
            observed = summary.get(field)
            if field == "n_samples":
                matches = observed == expected
            else:
                try:
                    matches = math.isclose(
                        float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
                    )
                except (TypeError, ValueError):
                    matches = False
            if not matches:
                raise ValueError(
                    f"dual adaptive summary differs from recomputation for {field}"
                )
        if summary.get("action_counts") != action_counts:
            raise ValueError("dual adaptive action counts do not recompute")
        reported_head_recall = summary.get("per_head_block_recall")
        if not isinstance(reported_head_recall, dict) or any(
            not math.isclose(
                float(reported_head_recall.get(name, -1.0)),
                head_blocks[name] / count,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name in ("text", "image")
        ):
            raise ValueError("dual adaptive per-head block recall does not recompute")
        if summary.get("decisive_head_counts") != decisive_counts:
            raise ValueError("dual adaptive decisive-head counts do not recompute")


def _load_feature_provenance(path: Path) -> dict[str, object]:
    required = {
        *PROVENANCE_KEYS,
        "id_column",
        "feature_source",
        "sample_ids",
        "text_embeddings",
        "image_embeddings",
        "attribution_features",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                "feature bundle lacks required provenance: "
                + ", ".join(sorted(missing))
            )
        scalar_keys = {*PROVENANCE_KEYS, "id_column", "feature_source"}
        result: dict[str, object] = {
            key: (
                int(np.asarray(data[key]).reshape(-1)[0])
                if key == "layer"
                else str(np.asarray(data[key]).reshape(-1)[0]).strip()
            )
            for key in scalar_keys
        }
        sample_ids = np.asarray(data["sample_ids"]).astype(str).reshape(-1)
        text = np.asarray(data["text_embeddings"], dtype=np.float64)
        image = np.asarray(data["image_embeddings"], dtype=np.float64)
        attribution = np.asarray(data["attribution_features"], dtype=np.float64)
        if (
            text.ndim != 2
            or image.ndim != 2
            or attribution.ndim != 2
            or text.shape != image.shape
            or len(text) != len(sample_ids)
            or len(attribution) != len(sample_ids)
            or not len(sample_ids)
            or not np.all(np.isfinite(text))
            or not np.all(np.isfinite(image))
            or not np.all(np.isfinite(attribution))
        ):
            raise ValueError(
                "feature arrays must be finite, aligned, non-empty 2D matrices; "
                "text and image dimensions must match"
            )
        if any(not value.strip() for value in sample_ids):
            raise ValueError("feature sample IDs must be non-empty")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("feature sample IDs must be unique")
        result.update(
            {
                "rows": int(len(sample_ids)),
                "text_feature_dim": int(text.shape[1]),
                "image_feature_dim": int(image.shape[1]),
                "attribution_feature_dim": int(attribution.shape[1]),
                "sample_ids_sha256": hashlib.sha256(
                    json.dumps(
                        sample_ids.tolist(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    for key in scalar_keys:
        value = result[key]
        if key != "layer" and not value:
            raise ValueError(f"feature provenance {key} must be non-empty")
    fingerprint = str(result["preprocessing_sha256"])
    if (
        len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("preprocessing_sha256 must be lowercase SHA-256 hex")
    return result


def _load_detector_context(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        if "threshold" not in data or "weights" not in data:
            raise ValueError("adaptive detector lacks threshold or weights")
        metadata = (
            json.loads(str(np.asarray(data["metadata_json"]).reshape(-1)[0]))
            if "metadata_json" in data
            else {}
        )
        if not isinstance(metadata, dict):
            raise ValueError("adaptive detector metadata_json must contain an object")
        if "feature_view" in metadata:
            feature_view = str(metadata["feature_view"])
        else:
            pooling = str(np.asarray(data["pooling"]).reshape(-1)[0])
            if pooling not in POOLING_FEATURE_VIEWS:
                raise ValueError(
                    f"cannot map detector pooling to feature view: {pooling!r}"
                )
            feature_view = POOLING_FEATURE_VIEWS[pooling]
        result: dict[str, object] = {
            "threshold": float(np.asarray(data["threshold"]).reshape(-1)[0]),
            "feature_dim": int(np.asarray(data["weights"]).size),
            "feature_view": feature_view,
        }
        if (
            not math.isfinite(float(result["threshold"]))
            or not 0.0 < float(result["threshold"]) < 1.0
            or int(result["feature_dim"]) <= 0
        ):
            raise ValueError("adaptive detector threshold or feature dimension is invalid")
        if "feature_source" in metadata:
            result["feature_source"] = str(metadata["feature_source"])
        for key in PROVENANCE_KEYS:
            if key in data:
                result[key] = (
                    int(np.asarray(data[key]).reshape(-1)[0])
                    if key == "layer"
                    else str(np.asarray(data[key]).reshape(-1)[0]).strip()
                )
            elif key in metadata:
                result[key] = (
                    int(metadata[key]) if key == "layer" else str(metadata[key])
                )
            else:
                raise ValueError(f"detector lacks required provenance: {key}")
    return result


def _validate_detector_feature_provenance(
    detector: dict[str, object],
    features: dict[str, object],
) -> None:
    mismatches: list[str] = []
    for key in PROVENANCE_KEYS:
        detector_value = detector[key]
        feature_value = features[key]
        if key == "model_id":
            detector_value = str(detector_value).replace("\\", "/")
            feature_value = str(feature_value).replace("\\", "/")
        if detector_value != feature_value:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "adaptive feature provenance does not match detector: "
            + ", ".join(mismatches)
        )


def _current_runtime(
    target: str,
) -> tuple[dict[str, object], DetectorArtifact | OrDetector, DeploymentConfig]:
    pyproject = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    version = next(
        line.split("=", 1)[1].strip().strip('"')
        for line in pyproject.splitlines()
        if line.startswith("version =")
    )
    config_path = REPOSITORY / "configs" / RUNTIME_CONFIGS[target]
    config = load_deployment_config(config_path)
    if config.target_profile != target:
        raise ValueError(
            f"runtime config target {config.target_profile!r} does not match {target!r}"
        )
    detector = load_configured_detector(config)
    _validate_runtime_provider_contract(config, detector)
    common = {
        key: getattr(detector, key)
        for key in COMMON_PROVENANCE_KEYS
    }
    base = {
        "aegis_version": version,
        "target_profile": config.target_profile,
        "traffic_mode": config.traffic_mode,
        "config": _portable_path(config_path),
        "config_sha256": _sha256(config_path),
        **common,
    }
    if isinstance(detector, OrDetector):
        configured_heads = {head.name: head for head in config.detector_heads}
        return (
            {
                **base,
                "detector_mode": "dual_or",
                "detector_identity_sha256": detector.identity_sha256,
                "pooling": detector.pooling,
                "feature_dim": detector.feature_dim,
                "base_feature_dim": detector.base_feature_dim,
                "composition": "Block > Review > Allow",
                "heads": [
                    {
                        "name": head.name,
                        "artifact": _portable_path(
                            configured_heads[head.name].artifact_path
                        ),
                        "artifact_sha256": head.artifact_sha256,
                        "source": head.artifact.source,
                        "pooling": head.artifact.pooling,
                        "feature_dim": head.artifact.feature_dim,
                        "block_threshold": float(head.artifact.threshold),
                        "review_threshold": float(head.review_threshold),
                    }
                    for head in detector.ordered_heads
                ],
            },
            detector,
            config,
        )
    assert config.detector_path is not None
    review_threshold = _legacy_review_threshold(config, detector)
    return (
        {
            **base,
            "detector_mode": "legacy_single_head",
            "detector": _portable_path(config.detector_path),
            "detector_sha256": str(detector.artifact_sha256),
            "detector_identity_sha256": str(detector.artifact_sha256),
            "source": detector.source,
            "pooling": detector.pooling,
            "feature_dim": detector.feature_dim,
            "block_threshold": float(detector.threshold),
            "review_threshold": review_threshold,
            "compatibility_note": (
                "Legacy schema-1 single-head compatibility mode; this is not the "
                "v7 dual-head OR architecture."
            ),
        },
        detector,
        config,
    )


def _validate_runtime_provider_contract(
    config: DeploymentConfig,
    detector: DetectorArtifact | OrDetector,
) -> None:
    expected = {
        "model_id": detector.model_id,
        "model_revision": detector.model_revision,
        "tokenizer_revision": detector.tokenizer_revision,
        "layer": detector.layer,
        "pooling": detector.pooling,
        "feature_dim": detector.feature_dim,
    }
    mismatches: list[str] = []
    for key, detector_value in expected.items():
        configured_value = config.provider_options.get(key)
        if key == "model_id":
            configured_value = _normalized_model_id(configured_value)
            detector_value = _normalized_model_id(detector_value)
        if configured_value != detector_value:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "runtime provider does not exactly match detector provenance/dimension: "
            + ", ".join(mismatches)
        )


def _legacy_review_threshold(
    config: DeploymentConfig,
    detector: DetectorArtifact,
) -> float:
    if config.policy.block_threshold not in (None, detector.threshold):
        raise ValueError(
            "legacy runtime block-threshold override is unsupported by this ablation"
        )
    if config.policy.review_threshold is None:
        raise ValueError(
            "legacy runtime scoring requires an explicit review_threshold; the "
            "historical symmetric uncertainty-margin policy is not representable "
            "as Block > Review > Allow"
        )
    review = float(config.policy.review_threshold)
    if not 0.0 <= review < detector.threshold:
        raise ValueError("legacy runtime review threshold is invalid")
    return review


def _score_runtime_detector(
    *,
    metadata: list[dict[str, str]],
    bundle: Any,
    feature_provenance: dict[str, object],
    detector: DetectorArtifact | OrDetector,
    config: DeploymentConfig,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    mismatches = _runtime_feature_mismatches(
        bundle=bundle,
        feature_provenance=feature_provenance,
        detector=detector,
    )
    if mismatches:
        return [], [], {
            "status": "not_scored_incompatible_features",
            "exact_compatibility_checks_passed": False,
            "mismatches": mismatches,
        }
    labels = np.asarray([int(row["label_id"]) for row in metadata], dtype=np.int64)
    if isinstance(detector, OrDetector):
        fused = np.concatenate(
            [bundle.text_embeddings, bundle.image_embeddings], axis=1
        )
        scores = detector.score_heads(fused)
        rows: list[dict[str, object]] = []
        component_actions: dict[str, list[str]] = {"text": [], "image": []}
        combined_actions: list[str] = []
        for index, metadata_row in enumerate(metadata):
            head_rows: dict[str, dict[str, object]] = {}
            for head in detector.ordered_heads:
                score = float(scores[head.name][index])
                action = _head_action(
                    score,
                    block_threshold=float(head.artifact.threshold),
                    review_threshold=float(head.review_threshold),
                )
                component_actions[head.name].append(action)
                head_rows[head.name] = {
                    "score": score,
                    "action": action,
                    "block": float(head.artifact.threshold),
                    "review": float(head.review_threshold),
                    "pooling": head.artifact.pooling,
                }
            decisive = _decisive_head(head_rows)
            combined = str(head_rows[decisive]["action"])
            combined_actions.append(combined)
            rows.append(
                {
                    **metadata_row,
                    "detector_mode": "dual_or",
                    "risk_score": head_rows[decisive]["score"],
                    "detector_threshold": head_rows[decisive]["block"],
                    "review_threshold": head_rows[decisive]["review"],
                    "verdict": "malicious" if combined == "block" else "benign",
                    "uncertain": combined == "review",
                    "recommended_action": combined,
                    "decisive_head": decisive,
                    "text_risk_score": head_rows["text"]["score"],
                    "text_block_threshold": head_rows["text"]["block"],
                    "text_review_threshold": head_rows["text"]["review"],
                    "text_action": head_rows["text"]["action"],
                    "text_pooling": head_rows["text"]["pooling"],
                    "image_risk_score": head_rows["image"]["score"],
                    "image_block_threshold": head_rows["image"]["block"],
                    "image_review_threshold": head_rows["image"]["review"],
                    "image_action": head_rows["image"]["action"],
                    "image_pooling": head_rows["image"]["pooling"],
                    "runtime_detector_identity_sha256": detector.identity_sha256,
                }
            )
        summaries = [
            _action_summary(labels, component_actions[name], name)
            for name in ("text", "image")
        ]
        summaries.append(_action_summary(labels, combined_actions, "combined_or"))
        return rows, summaries, {
            "status": "scored",
            "detector_mode": "dual_or",
            "rows": len(rows),
            "exact_compatibility_checks_passed": True,
            "composition": "Block > Review > Allow",
        }

    review = _legacy_review_threshold(config, detector)
    view_name = POOLING_FEATURE_VIEWS[detector.pooling]
    views = build_feature_views(
        bundle.text_embeddings,
        bundle.image_embeddings,
        bundle.attribution_features,
    )
    values = views[view_name]
    scores = detector.score(values)
    actions = [
        _head_action(
            float(score),
            block_threshold=float(detector.threshold),
            review_threshold=review,
        )
        for score in scores
    ]
    rows = [
        {
            **metadata_row,
            "detector_mode": "legacy_single_head",
            "risk_score": float(score),
            "detector_threshold": float(detector.threshold),
            "review_threshold": review,
            "verdict": "malicious" if action == "block" else "benign",
            "uncertain": action == "review",
            "recommended_action": action,
            "decisive_head": "legacy_single",
            "feature_view": view_name,
            "runtime_detector_identity_sha256": str(detector.artifact_sha256),
        }
        for metadata_row, score, action in zip(metadata, scores, actions)
    ]
    return rows, [_action_summary(labels, actions, "legacy_single_head")], {
        "status": "scored",
        "detector_mode": "legacy_single_head",
        "rows": len(rows),
        "exact_compatibility_checks_passed": True,
        "compatibility_note": (
            "Legacy schema-1 single-head compatibility mode; not a v7 OR pair."
        ),
    }


def _runtime_feature_mismatches(
    *,
    bundle: Any,
    feature_provenance: dict[str, object],
    detector: DetectorArtifact | OrDetector,
) -> list[str]:
    mismatches: list[str] = []
    for key in COMMON_PROVENANCE_KEYS:
        feature_value = feature_provenance[key]
        detector_value = getattr(detector, key)
        if key == "model_id":
            feature_value = _normalized_model_id(feature_value)
            detector_value = _normalized_model_id(detector_value)
        if feature_value != detector_value:
            mismatches.append(f"provenance:{key}")
    if feature_provenance["pooling"] != detector.pooling:
        mismatches.append("provenance:pooling")
    if isinstance(detector, OrDetector):
        expected = detector.base_feature_dim
        if bundle.text_embeddings.shape[1] != expected:
            mismatches.append("dimension:text_embeddings")
        if bundle.image_embeddings.shape[1] != expected:
            mismatches.append("dimension:image_embeddings")
        if expected * 2 != detector.feature_dim:
            mismatches.append("dimension:fused_detector")
    else:
        view = POOLING_FEATURE_VIEWS.get(detector.pooling)
        if view is None:
            mismatches.append("provenance:unsupported_pooling")
        else:
            views = build_feature_views(
                bundle.text_embeddings,
                bundle.image_embeddings,
                bundle.attribution_features,
            )
            if views[view].shape[1] != detector.feature_dim:
                mismatches.append(f"dimension:{view}")
    return sorted(set(mismatches))


def _head_action(
    score: float,
    *,
    block_threshold: float,
    review_threshold: float,
) -> str:
    if not math.isfinite(score):
        raise ValueError("detector score must be finite")
    if score >= block_threshold:
        return "block"
    if score >= review_threshold:
        return "review"
    return "allow"


def _decisive_head(heads: dict[str, dict[str, object]]) -> str:
    ranked: list[tuple[int, float, str]] = []
    for name, row in heads.items():
        action = str(row["action"])
        ranked.append((ACTION_RANK[action], _head_progress(row), name))
    winning_rank = max(item[0] for item in ranked)
    candidates = [item for item in ranked if item[0] == winning_rank]
    return sorted(candidates, key=lambda item: (-item[1], item[2]))[0][2]


def _head_progress(row: dict[str, object]) -> float:
    action = str(row["action"])
    score = float(row["score"])
    block = float(row["block"])
    review = float(row["review"])
    if action == "block":
        return (score - block) / (1.0 - block)
    if action == "review":
        return (score - review) / (block - review)
    return score / review if review > 0.0 else 0.0


def _action_summary(
    labels: np.ndarray,
    actions: list[str],
    component: str,
) -> dict[str, object]:
    values = np.asarray(actions, dtype=object)
    if len(values) != len(labels) or any(
        action not in ACTION_RANK for action in values
    ):
        raise ValueError("action rows are invalid or misaligned")
    malicious = labels == 1
    benign = labels == 0
    return {
        "component": component,
        "rows": int(len(values)),
        "allow": int(np.sum(values == "allow")),
        "review": int(np.sum(values == "review")),
        "block": int(np.sum(values == "block")),
        "malicious_rows": int(np.sum(malicious)),
        "malicious_blocked": int(np.sum((values == "block") & malicious)),
        "malicious_block_recall": float(np.mean(values[malicious] == "block")),
        "malicious_review_or_block_recall": float(
            np.mean(values[malicious] != "allow")
        ),
        "benign_rows": int(np.sum(benign)),
        "benign_blocked": int(np.sum((values == "block") & benign)),
        "benign_block_rate": float(np.mean(values[benign] == "block")),
        "benign_reviewed": int(np.sum((values == "review") & benign)),
    }


def _normalized_model_id(value: object) -> str:
    return str(value).replace("\\", "/")


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _reproduction_lines(
    *,
    runtime_target: str,
    metadata: Path,
    features: Path,
    output_dir: Path,
) -> list[str]:
    """Build a literal PowerShell reproduction block for this exact run."""
    if runtime_target not in RUNTIME_CONFIGS:
        raise ValueError(f"unsupported runtime target: {runtime_target}")

    unsafe = ("\r", "\n", "`", '"', "$")

    def validated_path(name: str, value: Path) -> tuple[Path, str]:
        raw = str(value)
        if raw.strip() in {"", ".", ".."}:
            raise ValueError(f"{name} path must identify a concrete location")
        if any(character in raw for character in unsafe):
            raise ValueError(
                f"{name} path contains a character unsafe for PowerShell"
            )
        resolved = value.expanduser().resolve()
        portable = _portable_path(resolved)
        if any(character in portable for character in unsafe):
            raise ValueError(
                f"{name} path contains a character unsafe for PowerShell"
            )
        return resolved, portable

    metadata_resolved, metadata_portable = validated_path("metadata", metadata)
    features_resolved, features_portable = validated_path("features", features)
    output_resolved, _ = validated_path("output directory", output_dir)
    run_root = output_resolved.parent
    run_root_portable = _portable_path(run_root)
    if any(character in run_root_portable for character in unsafe):
        raise ValueError("run root contains a character unsafe for PowerShell")

    def command_path(resolved: Path, portable: str) -> str:
        try:
            relative = resolved.relative_to(run_root)
        except ValueError:
            return portable
        if relative == Path("."):
            return "$run"
        return f"$run/{relative.as_posix()}"

    metadata_argument = command_path(metadata_resolved, metadata_portable)
    features_argument = command_path(features_resolved, features_portable)
    output_argument = command_path(output_resolved, _portable_path(output_resolved))
    return [
        "```powershell",
        f'$run = "{run_root_portable}"',
        "python deliverables/ablation_report/run_ablation.py `",
        f'  --metadata "{metadata_argument}" `',
        f'  --features "{features_argument}" `',
        f"  --runtime-target {runtime_target} `",
        f'  --output-dir "{output_argument}"',
        "```",
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_importance(results: list[dict[str, object]]) -> list[dict[str, object]]:
    all_row = next(row for row in results if row["feature_view"] == "all_input_signals")
    baseline_auroc = float(all_row["auroc"])
    baseline_auprc = float(all_row["auprc"])
    enriched: list[dict[str, object]] = []
    for row in results:
        enriched.append(
            {
                **row,
                "delta_auroc_vs_all": float(row["auroc"]) - baseline_auroc,
                "delta_auprc_vs_all": float(row["auprc"]) - baseline_auprc,
            }
        )
    return enriched


def _aggregate_low_label(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["trusted_per_class"]), int(row["pseudo_per_class"]))].append(row)
    summary: list[dict[str, object]] = []
    for (trusted, pseudo), group in sorted(groups.items()):
        item: dict[str, object] = {
            "trusted_per_class": trusted,
            "pseudo_per_class": pseudo,
            "seeds": len(group),
        }
        for metric in ("auroc", "auprc", "precision", "recall", "f1"):
            values = np.asarray([float(row[metric]) for row in group])
            item[f"mean_{metric}"] = float(values.mean())
            item[f"std_{metric}"] = float(values.std(ddof=0))
        item["mean_pseudo_selected"] = float(np.mean([int(row["n_pseudo"]) for row in group]))
        summary.append(item)
    return summary


def _attack_family_transfer(metadata: list[dict[str, str]], bundle) -> list[dict[str, object]]:
    labels = np.asarray([int(row["label_id"]) for row in metadata], dtype=np.int64)
    paired_styles = np.asarray([row["paired_attack_style"] for row in metadata])
    original_splits = np.asarray([row["split"] for row in metadata])
    views = build_feature_views(bundle.text_embeddings, bundle.image_embeddings, bundle.attribution_features)
    rows: list[dict[str, object]] = []
    for view_name in ("text_representation", "all_input_signals"):
        features = views[view_name]
        for held_style in sorted(set(paired_styles)):
            held = paired_styles == held_style
            train = (~held) & (original_splits == "train")
            validation = (~held) & (original_splits == "validation")
            transfer_split = np.full(len(metadata), "unused", dtype=object)
            transfer_split[train] = "train"
            transfer_split[validation] = "validation"
            transfer_split[held] = "test"
            outcome = fit_evaluate_view(features, labels, transfer_split, ExperimentConfig(seed=73))
            rows.append(
                {
                    "feature_view": view_name,
                    "held_out_attack_style": held_style,
                    "n_test": int(np.sum(held)),
                    **outcome["test_metrics"],
                }
            )
    return rows


def _uncertainty_coverage(metadata: list[dict[str, str]], fitted: dict[str, object]) -> list[dict[str, object]]:
    test_indices = np.asarray(fitted["test_indices"], dtype=np.int64)
    scores = np.asarray(fitted["test_scores"], dtype=np.float64)
    labels = np.asarray([int(metadata[index]["label_id"]) for index in test_indices])
    uncertainty = binary_entropy(scores)
    threshold = float(fitted["threshold"])
    predictions = scores >= threshold
    rows: list[dict[str, object]] = []
    for review_fraction in (0.0, 0.125, 0.25, 0.5):
        review_count = int(math.ceil(len(scores) * review_fraction))
        kept = np.ones(len(scores), dtype=bool)
        if review_count:
            kept[np.argsort(-uncertainty)[:review_count]] = False
        retained_accuracy = float(np.mean(predictions[kept] == labels[kept])) if np.any(kept) else float("nan")
        rows.append(
            {
                "review_fraction": review_fraction,
                "reviewed_rows": review_count,
                "retained_rows": int(np.sum(kept)),
                "coverage": float(np.mean(kept)),
                "retained_accuracy": retained_accuracy,
                "mean_retained_uncertainty": float(np.mean(uncertainty[kept])) if np.any(kept) else float("nan"),
                "threshold": threshold,
            }
        )
    return rows


def _runtime_description(runtime: dict[str, object]) -> str:
    if runtime["detector_mode"] == "dual_or":
        head_table = _markdown_table(
            list(runtime["heads"]),
            [
                "name",
                "pooling",
                "feature_dim",
                "block_threshold",
                "review_threshold",
                "artifact_sha256",
            ],
        )
        return (
            f"The current AEGIS `{runtime['aegis_version']}` "
            f"`{runtime['target_profile']}` profile uses a "
            f"{runtime['feature_dim']}-dimensional `{runtime['pooling']}` v7 "
            "dual-head detector. Text and image heads are scored independently; "
            "their actions are composed as `Block > Review > Allow`. The canonical "
            f"runtime detector identity is `{runtime['detector_identity_sha256']}`."
            "\n\n"
            + head_table
        )
    return (
        f"The current AEGIS `{runtime['aegis_version']}` "
        f"`{runtime['target_profile']}` profile uses `{runtime['detector']}` "
        f"(SHA-256 `{runtime['detector_sha256']}`), a "
        f"{runtime['feature_dim']}-dimensional `{runtime['pooling']}` detector with "
        f"block threshold `{runtime['block_threshold']}` and review threshold "
        f"`{runtime['review_threshold']}`. This is **legacy schema-1 single-head "
        "compatibility mode**, not the v7 dual-head OR architecture."
    )


def _render_report(
    full: list[dict[str, object]],
    low_label: list[dict[str, object]],
    transfer: list[dict[str, object]],
    uncertainty: list[dict[str, object]],
    adaptive: list[dict[str, object]],
    provenance: dict[str, object],
) -> str:
    status = str(provenance["scientific_status"])
    runtime = dict(provenance["runtime_context"])
    runtime_scoring = dict(provenance["runtime_scoring"])
    feature_provenance = dict(provenance["feature_provenance"])
    reproduction = dict(provenance["reproduction"])
    reproduction_command = reproduction.get("command")
    if not isinstance(reproduction_command, list) or not all(
        isinstance(line, str) for line in reproduction_command
    ):
        raise ValueError("reproduction command provenance is invalid")
    runtime_description = _runtime_description(runtime)
    if runtime_scoring["status"] == "scored":
        relationship = (
            "Exact model, revision, tokenizer, preprocessing, layer, pooling, and "
            "dimension checks passed. The following actions were recomputed from "
            "the supplied bundle."
        )
        runtime_score_table = _markdown_table(
            list(runtime_scoring["results"]),
            [
                "component",
                "allow",
                "review",
                "block",
                "malicious_rows",
                "malicious_blocked",
                "malicious_block_recall",
                "benign_rows",
                "benign_blocked",
                "benign_block_rate",
                "benign_reviewed",
            ],
        )
    else:
        relationship = (
            "The supplied research bundle was not scored by the runtime detector "
            "because exact compatibility checks failed: `"
            + "`, `".join(runtime_scoring["mismatches"])
            + "`. Signal-ablation results remain scoped only to their recorded "
            "research configuration."
        )
        runtime_score_table = "No compatible runtime-detector score table was produced."

    adaptive_evidence = provenance["adaptive_evidence"]
    if adaptive_evidence is None:
        adaptive_status = "No bounded-evasion panel was attached to this run."
        adaptive_table = "No scored adaptive panel was available for this run."
        adaptive_method = (
            "The optional evaluator operates over the first N fixed, deterministic "
            "variants and does not represent a sequential adaptive policy."
        )
    elif adaptive_evidence["detector_mode"] == "dual_or":
        adaptive_status = (
            "The attached panel is bound to pair identity `"
            f"{adaptive_evidence['pair_identity_sha256']}` and runtime detector "
            f"identity `{adaptive_evidence['runtime_detector_identity_sha256']}`."
        )
        adaptive_table = _markdown_table(
            adaptive,
            [
                "query_budget",
                "n_samples",
                "malicious_recall",
                "evasion_rate",
                "review_or_block_recall",
                "allow_rate",
                "review_rate",
                "block_rate",
                "mean_decisive_band_progress",
            ],
        )
        adaptive_method = (
            "For each budget, the dual-head evaluator selects the least restrictive "
            "aggregate action in hindsight, using the same `Block > Review > Allow` "
            "composition as runtime. It compares normalized progress only within "
            "the same action band; raw text-head and image-head scores are not "
            "treated as directly comparable."
        )
    else:
        adaptive_status = (
            "The attached panel uses explicitly labeled legacy single-head evidence "
            f"at block threshold `{adaptive_evidence['threshold']}`."
        )
        adaptive_table = _markdown_table(
            adaptive,
            [
                "query_budget",
                "n_samples",
                "threshold",
                "malicious_recall",
                "evasion_rate",
                "mean_worst_case_score",
            ],
        )
        adaptive_method = (
            "The legacy evaluator selects the lowest single-detector score in "
            "hindsight among the first N fixed variants."
        )
    lines = [
        "# AEGIS signal-importance and transferability ablation report",
        "",
        "## Evidence status",
        "",
        f"This run is classified as **`{status}`** and uses feature source "
        f"**`{provenance['feature_source']}`**. Its model is "
        f"`{feature_provenance['model_id']}` at model revision "
        f"`{feature_provenance['model_revision']}` and tokenizer revision "
        f"`{feature_provenance['tokenizer_revision']}`. The extraction uses layer "
        f"`{feature_provenance['layer']}`, pooling "
        f"`{feature_provenance['pooling']}`, and preprocessing fingerprint "
        f"`{feature_provenance['preprocessing_sha256']}`.",
        "",
        "## Relationship to the current runtime",
        "",
        runtime_description,
        "",
        relationship,
        "",
        runtime_score_table,
        "",
        "## Protocol",
        "",
        "Matched benign/adversarial groups remain in one split. Classifier fitting uses training "
        "rows, threshold selection uses validation rows, and the test split is not used for "
        "selection. The experiment detector is class-balanced logistic regression over pooled "
        "representations and compact signals.",
        "",
        "## Signal importance",
        "",
        _markdown_table(full, ["feature_view", "feature_dim", "auroc", "auprc", "precision", "recall", "f1", "mean_test_uncertainty"]),
        "",
        "`all_input_signals` combines text and image representations, cross-modal consistency, "
        "and perturbation-attribution summaries. Each comparison is scoped to the exact "
        "model and preprocessing provenance recorded above.",
        "",
        "## Low-label and pseudo-label ablation",
        "",
        _markdown_table(low_label, ["trusted_per_class", "pseudo_per_class", "seeds", "mean_auroc", "std_auroc", "mean_auprc", "std_auprc", "mean_f1", "mean_pseudo_selected"]),
        "",
        "Pseudo-labeling is accepted only when both predicted classes contribute the same number "
        "of high-confidence training rows. A zero `mean_pseudo_selected` records that the confidence "
        "gate correctly declined to invent supervision.",
        "",
        "## Attack-family transfer",
        "",
        _markdown_table(transfer, ["feature_view", "held_out_attack_style", "n_test", "auroc", "auprc", "precision", "recall", "f1"]),
        "",
        "Each transfer row holds out the matched groups of one attack family. Training and "
        "threshold selection use different attack families. Small held-out groups produce "
        "high-variance estimates and must not be generalized to deployment.",
        "",
        "## Uncertainty and review coverage",
        "",
        _markdown_table(uncertainty, ["review_fraction", "reviewed_rows", "retained_rows", "coverage", "retained_accuracy", "mean_retained_uncertainty"]),
        "",
        "Rows are ranked for review by normalized binary entropy. This table exposes the "
        "coverage/accuracy trade-off rather than treating near-threshold decisions as certain.",
        "",
        "## Bounded best-of-N detector evasion",
        "",
        adaptive_table,
        "",
        adaptive_method
        + " This is a bounded best-of-N oracle analysis of detector actions, not "
        "harmful response generation. "
        + adaptive_status,
        "",
        "## Conclusions",
        "",
        "The framework measures representation, modality, consistency, attribution, "
        "uncertainty, label-efficiency, and held-family transfer independently. Interpret every "
        "metric within the recorded dataset, model, preprocessing, threshold, and traffic-mode "
        "scope.",
        "",
        "## Reproduction",
        "",
        *reproduction_command,
        "",
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "nan" if math.isnan(value) else f"{value:.4f}"
            cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty result table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
