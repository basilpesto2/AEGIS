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

from AEGIS.evaluation import run_logistic_transfer_grid
from AEGIS.experiment_io import (
    load_feature_sets,
    require_column,
    require_metadata,
    unique_by_pooling,
)
from AEGIS.io import load_metadata
from AEGIS.labels import coerce_binary_labels
from AEGIS.logistic import LogisticRegressionNumpy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train supervised logistic detectors on one dataset and evaluate transfer "
            "on a separate target dataset."
        )
    )
    parser.add_argument("--source-embeddings", required=True, nargs="+")
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--target-embeddings", required=True, nargs="+")
    parser.add_argument("--target-metadata", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--target-score-csv",
        default=None,
        help="Optional per-target-sample score table for the selected source-validation setting.",
    )
    parser.add_argument("--source-name", default="source")
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    parser.add_argument("--l2-values", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--epochs-values", type=int, nargs="+", default=[300, 600])
    parser.add_argument("--malicious-prior", type=float, default=None)
    parser.add_argument("--selection-metric", default="auprc")
    parser.add_argument("--include-concat", action="store_true")
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="experiment_split")
    parser.add_argument("--source-fit-split", default="fit")
    parser.add_argument("--source-validation-split", default="val")
    parser.add_argument("--source-test-split", default="test")
    args = parser.parse_args()

    source_metadata = require_metadata(load_metadata(args.source_metadata), "source metadata")
    target_metadata = require_metadata(load_metadata(args.target_metadata), "target metadata")

    source_feature_sets = load_feature_sets(
        [Path(path) for path in args.source_embeddings],
        source_metadata,
        id_column=args.id_column,
    )
    target_feature_sets = load_feature_sets(
        [Path(path) for path in args.target_embeddings],
        target_metadata,
        id_column=args.id_column,
    )
    feature_pairs = _pair_feature_sets(source_feature_sets, target_feature_sets)
    if args.include_concat:
        feature_pairs.append(_concat_feature_pair(source_feature_sets, target_feature_sets))

    aligned_source_metadata = feature_pairs[0]["source_metadata"]
    aligned_target_metadata = feature_pairs[0]["target_metadata"]

    require_column(aligned_source_metadata, args.label_column, "source metadata")
    require_column(aligned_target_metadata, args.label_column, "target metadata")
    require_column(aligned_source_metadata, args.split_column, "source metadata")

    source_labels = coerce_binary_labels(aligned_source_metadata[args.label_column])
    target_labels = coerce_binary_labels(aligned_target_metadata[args.label_column])
    source_splits = aligned_source_metadata[args.split_column].astype(str).to_numpy()

    rows: list[dict] = []
    for pair in feature_pairs:
        results = run_logistic_transfer_grid(
            pair["source_features"],
            source_labels=source_labels,
            source_fit_mask=source_splits == args.source_fit_split,
            source_validation_mask=source_splits == args.source_validation_split,
            source_test_mask=source_splits == args.source_test_split,
            target_features=pair["target_features"],
            target_labels=target_labels,
            learning_rates=args.learning_rates,
            l2_values=args.l2_values,
            epochs_values=args.epochs_values,
            malicious_prior=args.malicious_prior,
        )

        for result in results:
            row = result.as_row()
            row.update(
                {
                    "feature_set": pair["name"],
                    "source_name": args.source_name,
                    "target_name": args.target_name,
                    "source_embeddings": pair["source_embeddings"],
                    "target_embeddings": pair["target_embeddings"],
                    "source_model_id": pair["source_model_id"],
                    "source_layer": pair["source_layer"],
                    "source_pooling": pair["source_pooling"],
                    "target_model_id": pair["target_model_id"],
                    "target_layer": pair["target_layer"],
                    "target_pooling": pair["target_pooling"],
                    "feature_dim": int(pair["source_features"].shape[1]),
                }
            )
            rows.append(row)

    if not rows:
        raise ValueError("No transfer results were produced.")

    metric_column = f"source_validation_{args.selection_metric}"
    if metric_column not in rows[0]:
        available = sorted(
            key.removeprefix("source_validation_")
            for key in rows[0]
            if key.startswith("source_validation_")
        )
        raise ValueError(
            f"Unknown selection metric {args.selection_metric!r}. Available metrics: {available}"
        )

    selected = max(rows, key=lambda row: row[metric_column])
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)

    target_score_output = None
    if args.target_score_csv is not None:
        selected_pair = _find_selected_pair(feature_pairs, selected)
        target_score_output = _write_selected_target_scores(
            output_path=Path(args.target_score_csv),
            selected=selected,
            source_features=selected_pair["source_features"],
            source_labels=source_labels,
            source_fit_mask=source_splits == args.source_fit_split,
            target_features=selected_pair["target_features"],
            target_metadata=aligned_target_metadata,
        )

    print(
        json.dumps(
            {
                "output": str(output),
                "target_score_output": target_score_output,
                "n_rows": len(rows),
                "selection_metric": args.selection_metric,
                "selected": selected,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _pair_feature_sets(source_sets: list[dict], target_sets: list[dict]) -> list[dict]:
    if len(source_sets) == 1 and len(target_sets) == 1:
        return [_make_feature_pair(source_sets[0], target_sets[0])]

    source_by_pooling = unique_by_pooling(source_sets, "source")
    target_by_pooling = unique_by_pooling(target_sets, "target")
    common_poolings = sorted(set(source_by_pooling) & set(target_by_pooling))
    if not common_poolings:
        raise ValueError("No common pooling values found between source and target embeddings.")

    return [
        _make_feature_pair(source_by_pooling[pooling], target_by_pooling[pooling])
        for pooling in common_poolings
    ]


def _concat_feature_pair(source_sets: list[dict], target_sets: list[dict]) -> dict:
    source_by_pooling = unique_by_pooling(source_sets, "source")
    target_by_pooling = unique_by_pooling(target_sets, "target")
    common_poolings = sorted(set(source_by_pooling) & set(target_by_pooling))
    if len(common_poolings) < 2:
        raise ValueError("At least two common pooling values are required for concat_all.")

    source_features = np.concatenate(
        [source_by_pooling[pooling]["features"] for pooling in common_poolings],
        axis=1,
    )
    target_features = np.concatenate(
        [target_by_pooling[pooling]["features"] for pooling in common_poolings],
        axis=1,
    )
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("Concatenated source and target features have different dimensions.")

    return {
        "name": "concat_all",
        "source_embeddings": ";".join(
            source_by_pooling[pooling]["embedding_file"] for pooling in common_poolings
        ),
        "target_embeddings": ";".join(
            target_by_pooling[pooling]["embedding_file"] for pooling in common_poolings
        ),
        "source_features": source_features,
        "target_features": target_features,
        "source_metadata": source_sets[0]["metadata"],
        "target_metadata": target_sets[0]["metadata"],
        "source_model_id": "mixed",
        "source_layer": None,
        "source_pooling": "concat_all",
        "target_model_id": "mixed",
        "target_layer": None,
        "target_pooling": "concat_all",
    }


def _make_feature_pair(source: dict, target: dict) -> dict:
    if source["features"].shape[1] != target["features"].shape[1]:
        raise ValueError(
            f"Feature dimension mismatch for {source['name']!r}: "
            f"source has {source['features'].shape[1]}, target has {target['features'].shape[1]}."
        )
    name = str(source["pooling"] or source["name"])
    return {
        "name": name,
        "source_embeddings": source["embedding_file"],
        "target_embeddings": target["embedding_file"],
        "source_features": source["features"],
        "target_features": target["features"],
        "source_metadata": source["metadata"],
        "target_metadata": target["metadata"],
        "source_model_id": source["model_id"],
        "source_layer": source["layer"],
        "source_pooling": source["pooling"],
        "target_model_id": target["model_id"],
        "target_layer": target["layer"],
        "target_pooling": target["pooling"],
    }


def _find_selected_pair(feature_pairs: list[dict], selected: dict) -> dict:
    for pair in feature_pairs:
        if pair["name"] == selected["feature_set"]:
            return pair
    raise ValueError(f"Could not find selected feature pair {selected['feature_set']!r}.")


def _write_selected_target_scores(
    output_path: Path,
    selected: dict,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    source_fit_mask: np.ndarray,
    target_features: np.ndarray,
    target_metadata: pd.DataFrame,
) -> str:
    classifier = LogisticRegressionNumpy(
        learning_rate=float(selected["learning_rate"]),
        l2=float(selected["l2"]),
        epochs=int(selected["epochs"]),
    ).fit(source_features[source_fit_mask], source_labels[source_fit_mask])

    probabilities = classifier.predict_proba(target_features)
    table = target_metadata.copy()
    table["transfer_malicious_probability"] = probabilities
    threshold = selected.get("threshold")
    if threshold is not None and not pd.isna(threshold):
        threshold = float(threshold)
        table["transfer_threshold"] = threshold
        table["predicted_malicious"] = (probabilities >= threshold).astype(int)
    table["transfer_source"] = selected["source_name"]
    table["transfer_learning_rate"] = float(selected["learning_rate"])
    table["transfer_l2"] = float(selected["l2"])
    table["transfer_epochs"] = int(selected["epochs"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return str(output_path)


if __name__ == "__main__":
    main()
