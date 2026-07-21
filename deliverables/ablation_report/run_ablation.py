from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DELIVERABLES = ROOT.parent
PIPELINE = DELIVERABLES / "reproducible_pipeline"
sys.path.insert(0, str(PIPELINE))

from aegis_research.experiment import ExperimentConfig, fit_evaluate_view, run_ablation_suite  # noqa: E402
from aegis_research.io import load_feature_bundle, load_metadata  # noqa: E402
from aegis_research.metrics import classification_metrics  # noqa: E402
from aegis_research.signals import binary_entropy, build_feature_views  # noqa: E402


DEFAULT_METADATA = DELIVERABLES / "annotated_benchmark" / "data" / "benchmark.csv"
DEFAULT_FEATURES = PIPELINE / "fixtures" / "smoke_features.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the AEGIS ablation and transfer report.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
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
    adaptive_path = DELIVERABLES / "red_teaming" / "generated" / "adaptive_summary.json"
    adaptive_rows: list[dict[str, object]] = []
    if adaptive_path.exists():
        adaptive_rows = list(json.loads(adaptive_path.read_text(encoding="utf-8"))["summary"])
        _write_csv(args.output_dir / "adaptive_redteam_summary.csv", adaptive_rows)

    provenance = {
        "metadata": str(args.metadata),
        "features": str(args.features),
        "feature_source": bundle.feature_source,
        "rows": len(metadata),
        "selection": "feature view and threshold use validation data only",
        "test_policy": "test rows are not used for hyperparameter or threshold selection",
        "scientific_status": (
            "software_smoke_test_only"
            if bundle.feature_source.startswith("deterministic_smoke")
            else "model_evidence_requires_manifest_review"
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = _render_report(full_rows, low_label_summary, transfer_rows, uncertainty_rows, adaptive_rows, provenance)
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(ROOT / "REPORT.md"), **provenance}, indent=2))


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
        "## Protocol",
        "",
        "Matched benign/adversarial groups remain in one split. Classifier fitting uses training "
        "rows, threshold selection uses validation rows, and the test split is not used for "
        "selection. The primary detector is class-balanced logistic regression over pooled "
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
        "## Bounded adaptive red-team evaluation",
        "",
        _markdown_table(adaptive, ["query_budget", "n_samples", "threshold", "malicious_recall", "evasion_rate", "mean_worst_case_score"]) if adaptive else "No scored adaptive panel was available for this run.",
        "",
        "The bounded adversary selects the lowest detector score among the first N deterministic "
        "variants. These smoke results intentionally reveal that the lightweight fixture detector "
        "is brittle; they measure detector evasion, not harmful MLLM response generation.",
        "",
        "## Legacy MLLM evidence",
        "",
        "`legacy_evidence.csv` preserves aggregate values reported in commit `3f6e46a`. The "
        "underlying processed data, embeddings, and output tables were ignored and are absent, "
        "so these values are historical context only—not independently rerun results.",
        "",
        "## Conclusions",
        "",
        "The recovered framework now measures representation, modality, consistency, attribution, "
        "uncertainty, label-efficiency, and held-family transfer independently. Scientific claims "
        "remain conditional on replacing the smoke fixture with provenance-complete MLLM features "
        "and rerunning this exact report.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python deliverables/annotated_benchmark/build_assets.py",
        "node deliverables/annotated_benchmark/build_workbook.mjs",
        "python deliverables/reproducible_pipeline/scripts/build_smoke_features.py",
        "python deliverables/ablation_report/run_ablation.py",
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
