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

import numpy as np


ROOT = Path(__file__).resolve().parent
DELIVERABLES = ROOT.parent
REPOSITORY = DELIVERABLES.parent
PIPELINE = DELIVERABLES / "pipeline"
sys.path.insert(0, str(PIPELINE))

from aegis_research.experiment import ExperimentConfig, fit_evaluate_view, run_ablation_suite  # noqa: E402
from aegis_research.io import load_feature_bundle, load_metadata  # noqa: E402
from aegis_research.metrics import classification_metrics  # noqa: E402
from aegis_research.signals import binary_entropy, build_feature_views  # noqa: E402


DEFAULT_METADATA = DELIVERABLES / "benchmark" / "data" / "benchmark.csv"
DEFAULT_FEATURES = PIPELINE / "fixtures" / "smoke_features.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the AEGIS ablation and transfer report.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "reproduction_v1",
        help="Versioned output directory; committed historical results are never overwritten by default.",
    )
    parser.add_argument("--report", type=Path, help="Report path; defaults to OUTPUT_DIR/REPORT.md.")
    parser.add_argument("--adaptive-summary", type=Path)
    parser.add_argument("--adaptive-scores", type=Path)
    parser.add_argument("--adaptive-detector", type=Path)
    parser.add_argument("--adaptive-features", type=Path)
    parser.add_argument("--force", action="store_true", help="Allow replacement of an existing run directory.")
    args = parser.parse_args()
    orphaned_adaptive = [
        name
        for name, value in (
            ("--adaptive-scores", args.adaptive_scores),
            ("--adaptive-detector", args.adaptive_detector),
            ("--adaptive-features", args.adaptive_features),
        )
        if value is not None and args.adaptive_summary is None
    ]
    if orphaned_adaptive:
        parser.error(
            "--adaptive-summary is required when using "
            + ", ".join(orphaned_adaptive)
        )
    report_path = args.report or args.output_dir / "REPORT.md"
    _prepare_output(args.output_dir, report_path, force=args.force)
    metadata = load_metadata(args.metadata)
    bundle = load_feature_bundle(args.features, metadata)
    labels = np.asarray([int(row["label_id"]) for row in metadata], dtype=np.int64)
    splits = np.asarray([row["split"] for row in metadata])
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        "rows": len(metadata),
        "selection": "feature view and threshold use validation data only",
        "test_policy": "test rows are not used for hyperparameter or threshold selection",
        "scientific_status": (
            "software_smoke_test_only"
            if bundle.feature_source.startswith("deterministic_smoke")
            else "model_evidence_requires_manifest_review"
        ),
        "adaptive_evidence": adaptive_provenance,
        "runtime_context": _current_runtime_context(),
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
        "--adaptive-detector": args.adaptive_detector,
        "--adaptive-features": args.adaptive_features,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(
            "--adaptive-summary requires " + ", ".join(sorted(missing))
        )
    payload = json.loads(args.adaptive_summary.read_text(encoding="utf-8"))
    rows = list(payload.get("summary", []))
    if not rows:
        raise ValueError("adaptive summary does not contain result rows")
    with np.load(args.adaptive_detector, allow_pickle=False) as detector:
        detector_threshold = float(np.asarray(detector["threshold"]).reshape(-1)[0])
        metadata_json = json.loads(
            str(np.asarray(detector["metadata_json"]).reshape(-1)[0])
        )
    with np.load(args.adaptive_features, allow_pickle=False) as features:
        feature_source = str(np.asarray(features["feature_source"]).reshape(-1)[0])
    summary_thresholds = [float(row["threshold"]) for row in rows]
    if any(
        not math.isclose(value, detector_threshold, rel_tol=0.0, abs_tol=1e-12)
        for value in summary_thresholds
    ):
        raise ValueError(
            "adaptive summary threshold does not match the supplied fixture detector"
        )
    with args.adaptive_scores.open("r", encoding="utf-8-sig", newline="") as handle:
        score_rows = list(csv.DictReader(handle))
    if not score_rows or any(row.get("label_id") != "1" for row in score_rows):
        raise ValueError("adaptive scores must be a non-empty malicious-only panel")
    score_thresholds = [_recorded_detector_threshold(row) for row in score_rows]
    if any(value is None for value in score_thresholds) or any(
        not math.isclose(
            float(value),
            detector_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for value in score_thresholds
    ):
        raise ValueError("adaptive score thresholds do not match the detector")
    score_views = {row["feature_view"] for row in score_rows}
    if score_views != {str(metadata_json["feature_view"])}:
        raise ValueError("adaptive score feature view does not match the detector")
    return rows, {
        "summary": _portable_path(args.adaptive_summary),
        "summary_sha256": _sha256(args.adaptive_summary),
        "scores": _portable_path(args.adaptive_scores),
        "scores_sha256": _sha256(args.adaptive_scores),
        "detector": _portable_path(args.adaptive_detector),
        "detector_sha256": _sha256(args.adaptive_detector),
        "features": _portable_path(args.adaptive_features),
        "features_sha256": _sha256(args.adaptive_features),
        "feature_source": feature_source,
        "feature_view": str(metadata_json["feature_view"]),
        "threshold": detector_threshold,
        "threshold_kind": "fixture_detector_block_threshold",
        "selection_policy": "bounded best-of-N over a fixed ordered variant panel",
    }


def _current_runtime_context() -> dict[str, object]:
    pyproject = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    version = next(
        line.split("=", 1)[1].strip().strip('"')
        for line in pyproject.splitlines()
        if line.startswith("version =")
    )
    config_path = REPOSITORY / "configs" / "aegis.llava.deployment.container.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    detector_path = REPOSITORY / str(config["detector"])
    with np.load(detector_path, allow_pickle=False) as detector:
        detector_context = {
            "source": str(np.asarray(detector["source"]).reshape(-1)[0]),
            "model_family": str(np.asarray(detector["model_family"]).reshape(-1)[0]),
            "model_id": str(np.asarray(detector["model_id"]).reshape(-1)[0]),
            "model_revision": str(
                np.asarray(detector["model_revision"]).reshape(-1)[0]
            ),
            "tokenizer_revision": str(
                np.asarray(detector["tokenizer_revision"]).reshape(-1)[0]
            ),
            "preprocessing_sha256": str(
                np.asarray(detector["preprocessing_sha256"]).reshape(-1)[0]
            ),
            "pooling": str(np.asarray(detector["pooling"]).reshape(-1)[0]),
            "feature_dim": int(np.asarray(detector["weights"]).size),
            "block_threshold": float(
                np.asarray(detector["threshold"]).reshape(-1)[0]
            ),
        }
    return {
        "aegis_version": version,
        "target_profile": config["target_profile"],
        "traffic_mode": config["traffic_mode"],
        "detector": config["detector"],
        "detector_sha256": _sha256(detector_path),
        "review_threshold": config["policy"]["review_threshold"],
        **detector_context,
    }


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


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
    adaptive_status = (
        "No adaptive fixture panel was attached to this run."
        if provenance["adaptive_evidence"] is None
        else (
            "The attached panel uses the 8-dimensional fixture detector at threshold "
            f"`{provenance['adaptive_evidence']['threshold']}`; it does not use the "
            "deployed LLaVA detector."
        )
    )
    lines = [
        "# AEGIS signal-importance and transferability ablation report",
        "",
        "## Evidence status",
        "",
        f"This run is classified as **`{status}`** and uses feature source "
        f"**`{provenance['feature_source']}`**. The deterministic fixture verifies the complete "
        "training, validation, thresholding, low-label, uncertainty, transfer, and reporting path. "
        "It is not evidence of MLLM safety performance.",
        "",
        "## Relationship to the current runtime",
        "",
        f"The current AEGIS `{runtime['aegis_version']}` LLaVA profile uses "
        f"`{runtime['detector']}` (SHA-256 `{runtime['detector_sha256']}`), an "
        f"{runtime['feature_dim']}-dimensional `{runtime['pooling']}` detector with block "
        f"threshold `{runtime['block_threshold']}` and review threshold "
        f"`{runtime['review_threshold']}`. The tables below use a separate deterministic "
        "fixture and must not be cited as tuned-v3 accuracy, attack-success, or robustness "
        "measurements. Current functional runtime evidence is recorded in "
        "`deliverables/runtime_validation/`.",
        "",
        "## Protocol",
        "",
        "Matched benign/adversarial groups remain in one split. Classifier fitting uses training "
        "rows, threshold selection uses validation rows, and the test split is not used for "
        "selection. The fixture experiment detector is class-balanced logistic regression over pooled "
        "representations and compact signals.",
        "",
        "## Signal importance",
        "",
        _markdown_table(full, ["feature_view", "feature_dim", "auroc", "auprc", "precision", "recall", "f1", "mean_test_uncertainty"]),
        "",
        "`all_input_signals` combines text and image representations, cross-modal consistency, "
        "and perturbation-attribution summaries. Comparisons are diagnostic because the smoke "
        "extractor is intentionally lightweight and its lexical cues are not MLLM hidden states.",
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
        "Each transfer row holds out all four matched groups of one attack family. Training and "
        "threshold selection use different attack families. With only eight held-out rows per "
        "family, these estimates have high variance and must not be generalized to deployment.",
        "",
        "## Uncertainty and review coverage",
        "",
        _markdown_table(uncertainty, ["review_fraction", "reviewed_rows", "retained_rows", "coverage", "retained_accuracy", "mean_retained_uncertainty"]),
        "",
        "Rows are ranked for review by normalized binary entropy. This table exposes the "
        "coverage/accuracy trade-off rather than treating near-threshold decisions as certain.",
        "",
        "## Bounded best-of-N fixture evasion",
        "",
        _markdown_table(adaptive, ["query_budget", "n_samples", "threshold", "malicious_recall", "evasion_rate", "mean_worst_case_score"]) if adaptive else "No scored adaptive panel was available for this run.",
        "",
        "The evaluator selects the lowest detector score in hindsight among the first N fixed, "
        "deterministic variants. This is a bounded best-of-N oracle analysis, not a sequential "
        "adaptive attack policy. These smoke results measure fixture-detector evasion, not harmful "
        "MLLM response generation. " + adaptive_status,
        "",
        "## Legacy MLLM evidence",
        "",
        "`legacy_evidence.csv` preserves aggregate values reported in commit `3f6e46a`. The "
        "underlying processed data, embeddings, and output tables were ignored and are absent, "
        "so these values are historical context only - not independently rerun results.",
        "",
        "## Conclusions",
        "",
        "The framework measures representation, modality, consistency, attribution, "
        "uncertainty, label-efficiency, and held-family transfer independently. Scientific claims "
        "remain conditional on replacing the smoke fixture with provenance-complete MLLM features "
        "and rerunning this report into a new versioned output directory.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        '$run = "deliverables/reproductions/ablation_v1"',
        "",
        "python deliverables/pipeline/scripts/build_smoke_features.py `",
        '  --output "$run/benchmark_smoke_features.npz"',
        "python deliverables/ablation_report/run_ablation.py `",
        '  --features "$run/benchmark_smoke_features.npz" `',
        '  --output-dir "$run/results"',
        "```",
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
        raise ValueError("adaptive score row contains conflicting threshold aliases")
    return values[0]


if __name__ == "__main__":
    main()
