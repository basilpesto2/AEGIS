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
PROVENANCE_KEYS = (
    "model_family",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "preprocessing_sha256",
    "layer",
    "pooling",
)


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
    feature_provenance = _load_feature_provenance(args.features)
    if feature_provenance["id_column"] != "sample_id":
        raise ValueError("ablation requires a feature bundle keyed by sample_id")
    metadata = load_metadata(args.metadata)
    bundle = load_feature_bundle(args.features, metadata)
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
        "feature_provenance": feature_provenance,
        "rows": len(metadata),
        "selection": "feature view and threshold use validation data only",
        "test_policy": "test rows are not used for hyperparameter or threshold selection",
        "scientific_status": "provenance_complete_model_evidence",
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
    detector_context = _load_detector_context(args.adaptive_detector)
    detector_threshold = float(detector_context["threshold"])
    adaptive_feature_provenance = _load_feature_provenance(
        args.adaptive_features
    )
    _validate_detector_feature_provenance(
        detector_context,
        adaptive_feature_provenance,
    )
    summary_thresholds = [float(row["threshold"]) for row in rows]
    if any(
        not math.isclose(value, detector_threshold, rel_tol=0.0, abs_tol=1e-12)
        for value in summary_thresholds
    ):
        raise ValueError(
            "adaptive summary threshold does not match the supplied detector"
        )
    with args.adaptive_scores.open("r", encoding="utf-8-sig", newline="") as handle:
        score_rows = list(csv.DictReader(handle))
    if not score_rows or any(row.get("label_id") != "1" for row in score_rows):
        raise ValueError("adaptive scores must be a non-empty malicious-only panel")
    if "detector_threshold" not in score_rows[0]:
        raise ValueError("adaptive scores must record detector_threshold")
    score_thresholds = [float(row["detector_threshold"]) for row in score_rows]
    if any(
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
    if score_views != {str(detector_context["feature_view"])}:
        raise ValueError("adaptive score feature view does not match the detector")
    threshold_kind = str(summary_provenance.get("threshold_kind", ""))
    if threshold_kind != "detector_artifact_block_threshold":
        raise ValueError(
            "ablation import requires detector_artifact_block_threshold evidence"
        )
    return rows, {
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
        "selection_policy": "bounded best-of-N over a fixed ordered variant panel",
    }


def _load_feature_provenance(path: Path) -> dict[str, object]:
    required = {*PROVENANCE_KEYS, "id_column", "feature_source"}
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                "feature bundle lacks required provenance: "
                + ", ".join(sorted(missing))
            )
        result: dict[str, object] = {
            key: (
                int(np.asarray(data[key]).reshape(-1)[0])
                if key == "layer"
                else str(np.asarray(data[key]).reshape(-1)[0]).strip()
            )
            for key in required
        }
    for key, value in result.items():
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
        if "feature_view" in metadata:
            feature_view = str(metadata["feature_view"])
        else:
            pooling = str(np.asarray(data["pooling"]).reshape(-1)[0])
            feature_views = {
                "text_tokens": "text_representation",
                "image_tokens": "image_representation",
            }
            if pooling not in feature_views:
                raise ValueError(
                    f"cannot map detector pooling to feature view: {pooling!r}"
                )
            feature_view = feature_views[pooling]
        result: dict[str, object] = {
            "threshold": float(np.asarray(data["threshold"]).reshape(-1)[0]),
            "feature_dim": int(np.asarray(data["weights"]).size),
            "feature_view": feature_view,
        }
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
            "layer": int(np.asarray(detector["layer"]).reshape(-1)[0]),
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
    feature_provenance = dict(provenance["feature_provenance"])
    runtime_match = all(
        (
            str(feature_provenance[key]).replace("\\", "/")
            if key == "model_id"
            else feature_provenance[key]
        )
        == (
            str(runtime[key]).replace("\\", "/")
            if key == "model_id"
            else runtime[key]
        )
        for key in PROVENANCE_KEYS
    )
    relationship = (
        "The feature bundle matches the deployed detector's model, tokenizer, "
        "layer, pooling, and preprocessing provenance."
        if runtime_match
        else (
            "The feature bundle does not match every deployed-detector provenance "
            "field; its results apply only to the recorded research configuration."
        )
    )
    adaptive_status = (
        "No bounded-evasion panel was attached to this run."
        if provenance["adaptive_evidence"] is None
        else (
            "The attached panel uses the recorded detector and feature provenance "
            f"at threshold `{provenance['adaptive_evidence']['threshold']}`."
        )
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
        f"The current AEGIS `{runtime['aegis_version']}` LLaVA profile uses "
        f"`{runtime['detector']}` (SHA-256 `{runtime['detector_sha256']}`), an "
        f"{runtime['feature_dim']}-dimensional `{runtime['pooling']}` detector with block "
        f"threshold `{runtime['block_threshold']}` and review threshold "
        f"`{runtime['review_threshold']}`. {relationship}",
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
        _markdown_table(adaptive, ["query_budget", "n_samples", "threshold", "malicious_recall", "evasion_rate", "mean_worst_case_score"]) if adaptive else "No scored adaptive panel was available for this run.",
        "",
        "The evaluator selects the lowest detector score in hindsight among the first N fixed, "
        "deterministic variants. This is a bounded best-of-N oracle analysis, not a sequential "
        "adaptive attack policy. It measures detector-score evasion, not harmful response "
        "generation. " + adaptive_status,
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
        "```powershell",
        '$run = "deliverables/runs/current_llava"',
        "python deliverables/ablation_report/run_ablation.py `",
        '  --metadata "$run/features/aligned_source_metadata.csv" `',
        '  --features "$run/features/feature_bundle.npz" `',
        '  --output-dir "$run/ablation"',
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


if __name__ == "__main__":
    main()
