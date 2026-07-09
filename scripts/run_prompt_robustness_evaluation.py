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
        description="Train on original prompts and evaluate unseen transformed malicious prompts."
    )
    parser.add_argument("--original-embeddings", required=True)
    parser.add_argument("--original-metadata", required=True)
    parser.add_argument("--transformed-embeddings", required=True)
    parser.add_argument("--transformed-metadata", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--breakdown-output-csv")
    parser.add_argument("--variant-column", default="robustness_variant")
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
    parser.add_argument("--selection-metric", default="auprc")
    args = parser.parse_args()

    original_metadata = _require_metadata(load_metadata(args.original_metadata), "original")
    transformed_metadata = _require_metadata(load_metadata(args.transformed_metadata), "transformed")
    original_x, original_metadata = _load_aligned(
        args.original_embeddings,
        original_metadata,
    )
    transformed_x, transformed_metadata = _load_aligned(
        args.transformed_embeddings,
        transformed_metadata,
    )

    for column in [args.label_column, args.split_column]:
        _require_column(original_metadata, column, "original metadata")
    _require_column(transformed_metadata, args.variant_column, "transformed metadata")
    for metadata, name in [
        (original_metadata, "original metadata"),
        (transformed_metadata, "transformed metadata"),
    ]:
        _require_column(metadata, args.family_column, name)

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
    metric_column = f"validation_{args.selection_metric}"
    grid_rows = [result.as_row() for result in grid]
    selected = max(grid_rows, key=lambda row: row[metric_column])

    classifier = LogisticRegressionNumpy(
        learning_rate=float(selected["learning_rate"]),
        l2=float(selected["l2"]),
        epochs=int(selected["epochs"]),
    ).fit(original_x[fit_mask], labels[fit_mask])
    threshold = float(selected["threshold"])

    rows = []
    breakdown_rows = []
    baseline_scores = classifier.predict_proba(original_x[test_mask])
    baseline_predictions = predict_from_threshold(baseline_scores, threshold)
    rows.append(
        _result_row(
            variant="original",
            labels=labels[test_mask],
            scores=baseline_scores,
            predictions=baseline_predictions,
            selected=selected,
        )
    )

    benign_test_mask = test_mask & (labels == 0)
    benign_features = original_x[benign_test_mask]
    benign_labels = labels[benign_test_mask]
    original_families = original_metadata[args.family_column].astype(str).to_numpy()
    transformed_families = transformed_metadata[args.family_column].astype(str).to_numpy()
    families = sorted(transformed_metadata[args.family_column].astype(str).unique())
    for family in families:
        family_mask = test_mask & (labels == 1) & (original_families == family)
        breakdown_rows.append(
            _breakdown_result_row(
                variant="original",
                family=family,
                malicious_features=original_x[family_mask],
                benign_features=benign_features,
                benign_labels=benign_labels,
                classifier=classifier,
                threshold=threshold,
                selected=selected,
            )
        )
    variants = sorted(transformed_metadata[args.variant_column].astype(str).unique())
    for variant in variants:
        variant_mask = transformed_metadata[args.variant_column].astype(str).to_numpy() == variant
        variant_features = transformed_x[variant_mask]
        target_features = np.concatenate([variant_features, benign_features], axis=0)
        target_labels = np.concatenate(
            [np.ones(len(variant_features), dtype=np.int64), benign_labels],
            axis=0,
        )
        scores = classifier.predict_proba(target_features)
        predictions = predict_from_threshold(scores, threshold)
        rows.append(
            _result_row(
                variant=variant,
                labels=target_labels,
                scores=scores,
                predictions=predictions,
                selected=selected,
            )
        )
        for family in families:
            family_variant_mask = variant_mask & (transformed_families == family)
            breakdown_rows.append(
                _breakdown_result_row(
                    variant=variant,
                    family=family,
                    malicious_features=transformed_x[family_variant_mask],
                    benign_features=benign_features,
                    benign_labels=benign_labels,
                    classifier=classifier,
                    threshold=threshold,
                    selected=selected,
                )
            )

    output = pd.DataFrame(rows)
    baseline = output.loc[output["variant"].eq("original")].iloc[0]
    output["auroc_delta_vs_original"] = output["auroc"] - baseline["auroc"]
    output["auprc_delta_vs_original"] = output["auprc"] - baseline["auprc"]
    output["f1_delta_vs_original"] = output["f1"] - baseline["f1"]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    breakdown_output = pd.DataFrame(breakdown_rows)
    breakdown_path = (
        Path(args.breakdown_output_csv)
        if args.breakdown_output_csv
        else output_path.with_name(f"{output_path.stem}_by_family.csv")
    )
    breakdown_output.to_csv(breakdown_path, index=False)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "breakdown_output": str(breakdown_path),
                "selected": selected,
                "results": output.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _breakdown_result_row(
    variant: str,
    family: str,
    malicious_features: np.ndarray,
    benign_features: np.ndarray,
    benign_labels: np.ndarray,
    classifier: LogisticRegressionNumpy,
    threshold: float,
    selected: dict,
) -> dict:
    target_features = np.concatenate([malicious_features, benign_features], axis=0)
    target_labels = np.concatenate(
        [np.ones(len(malicious_features), dtype=np.int64), benign_labels],
        axis=0,
    )
    scores = classifier.predict_proba(target_features)
    predictions = predict_from_threshold(scores, threshold)
    row = _result_row(
        variant=variant,
        labels=target_labels,
        scores=scores,
        predictions=predictions,
        selected=selected,
    )
    row["family"] = family
    row["n_malicious"] = int(len(malicious_features))
    row["n_benign"] = int(len(benign_features))
    return row


def _result_row(
    variant: str,
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    selected: dict,
) -> dict:
    row = {
        "variant": variant,
        "n_samples": int(len(labels)),
        "positive_rate": float(np.mean(predictions)),
        "threshold": float(selected["threshold"]),
        "learning_rate": float(selected["learning_rate"]),
        "l2": float(selected["l2"]),
        "epochs": int(selected["epochs"]),
    }
    row.update(detection_report(labels, scores, predictions))
    return row


def _load_aligned(path: str, metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings, sample_ids = load_embeddings(path)
    aligned, aligned_metadata, _ = align_embeddings_with_metadata(
        embeddings,
        sample_ids,
        metadata,
    )
    if aligned_metadata is None:
        raise ValueError("Metadata alignment failed.")
    return aligned, aligned_metadata


def _require_metadata(metadata: pd.DataFrame | None, name: str) -> pd.DataFrame:
    if metadata is None:
        raise ValueError(f"{name} metadata is required.")
    return metadata


def _require_column(metadata: pd.DataFrame, column: str, name: str) -> None:
    if column not in metadata.columns:
        raise ValueError(f"{name} is missing required column {column!r}.")


if __name__ == "__main__":
    main()
