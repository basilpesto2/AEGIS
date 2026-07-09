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

from AEGIS.evaluation import predict_from_threshold, threshold_from_prior
from AEGIS.io import align_embeddings_with_metadata, load_embeddings, load_metadata
from AEGIS.labels import coerce_binary_labels
from AEGIS.logistic import LogisticRegressionNumpy
from AEGIS.metrics import detection_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train on one labeled source, select hyperparameters on a different "
            "labeled source, and evaluate transfer on a target dataset."
        )
    )
    parser.add_argument("--train-embeddings", required=True, nargs="+")
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--validation-embeddings", required=True, nargs="+")
    parser.add_argument("--validation-metadata", required=True)
    parser.add_argument("--target-embeddings", required=True, nargs="+")
    parser.add_argument("--target-metadata", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--target-score-csv", default=None)
    parser.add_argument("--train-name", default="train_source")
    parser.add_argument("--validation-name", default="validation_source")
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
    parser.add_argument("--train-fit-split", default="fit")
    parser.add_argument("--train-test-split", default="test")
    parser.add_argument("--validation-split", default="val")
    args = parser.parse_args()

    train_metadata = _require_metadata(load_metadata(args.train_metadata), "train")
    validation_metadata = _require_metadata(load_metadata(args.validation_metadata), "validation")
    target_metadata = _require_metadata(load_metadata(args.target_metadata), "target")

    train_sets = _load_feature_sets(
        [Path(path) for path in args.train_embeddings],
        train_metadata,
        id_column=args.id_column,
    )
    validation_sets = _load_feature_sets(
        [Path(path) for path in args.validation_embeddings],
        validation_metadata,
        id_column=args.id_column,
    )
    target_sets = _load_feature_sets(
        [Path(path) for path in args.target_embeddings],
        target_metadata,
        id_column=args.id_column,
    )

    feature_triples = _make_feature_triples(train_sets, validation_sets, target_sets)
    if args.include_concat:
        feature_triples.append(_concat_feature_triple(train_sets, validation_sets, target_sets))

    train_aligned_metadata = feature_triples[0]["train_metadata"]
    validation_aligned_metadata = feature_triples[0]["validation_metadata"]
    target_aligned_metadata = feature_triples[0]["target_metadata"]

    _require_column(train_aligned_metadata, args.label_column, "train metadata")
    _require_column(validation_aligned_metadata, args.label_column, "validation metadata")
    _require_column(target_aligned_metadata, args.label_column, "target metadata")
    _require_column(train_aligned_metadata, args.split_column, "train metadata")
    _require_column(validation_aligned_metadata, args.split_column, "validation metadata")

    train_labels = coerce_binary_labels(train_aligned_metadata[args.label_column])
    validation_labels = coerce_binary_labels(validation_aligned_metadata[args.label_column])
    target_labels = coerce_binary_labels(target_aligned_metadata[args.label_column])
    train_splits = train_aligned_metadata[args.split_column].astype(str).to_numpy()
    validation_splits = validation_aligned_metadata[args.split_column].astype(str).to_numpy()

    train_fit_mask = train_splits == args.train_fit_split
    train_test_mask = train_splits == args.train_test_split
    validation_mask = validation_splits == args.validation_split
    if not np.any(train_fit_mask):
        raise ValueError(f"No train rows found for split {args.train_fit_split!r}.")
    if not np.any(train_test_mask):
        raise ValueError(f"No train rows found for split {args.train_test_split!r}.")
    if not np.any(validation_mask):
        raise ValueError(f"No validation rows found for split {args.validation_split!r}.")

    rows: list[dict] = []
    for triple in feature_triples:
        rows.extend(
            _run_grid(
                triple=triple,
                train_labels=train_labels,
                validation_labels=validation_labels,
                target_labels=target_labels,
                train_fit_mask=train_fit_mask,
                train_test_mask=train_test_mask,
                validation_mask=validation_mask,
                learning_rates=args.learning_rates,
                l2_values=args.l2_values,
                epochs_values=args.epochs_values,
                malicious_prior=args.malicious_prior,
                train_name=args.train_name,
                validation_name=args.validation_name,
                target_name=args.target_name,
            )
        )

    if not rows:
        raise ValueError("No cross-source transfer results were produced.")

    metric_column = f"validation_{args.selection_metric}"
    if metric_column not in rows[0]:
        available = sorted(
            key.removeprefix("validation_")
            for key in rows[0]
            if key.startswith("validation_")
        )
        raise ValueError(
            f"Unknown selection metric {args.selection_metric!r}. Available metrics: {available}"
        )

    selected = max(rows, key=lambda row: row[metric_column])
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    target_score_output = None
    if args.target_score_csv is not None:
        selected_triple = _find_selected_triple(feature_triples, selected["feature_set"])
        target_score_output = _write_selected_target_scores(
            output_path=Path(args.target_score_csv),
            selected=selected,
            train_features=selected_triple["train_features"],
            train_labels=train_labels,
            train_fit_mask=train_fit_mask,
            target_features=selected_triple["target_features"],
            target_metadata=target_aligned_metadata,
        )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "target_score_output": target_score_output,
                "n_rows": len(rows),
                "selection_metric": args.selection_metric,
                "selected": selected,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_grid(
    triple: dict,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    target_labels: np.ndarray,
    train_fit_mask: np.ndarray,
    train_test_mask: np.ndarray,
    validation_mask: np.ndarray,
    learning_rates: list[float],
    l2_values: list[float],
    epochs_values: list[int],
    malicious_prior: float | None,
    train_name: str,
    validation_name: str,
    target_name: str,
) -> list[dict]:
    rows = []
    train_x = triple["train_features"]
    validation_x = triple["validation_features"]
    target_x = triple["target_features"]
    for learning_rate in learning_rates:
        for l2 in l2_values:
            for epochs in epochs_values:
                classifier = LogisticRegressionNumpy(
                    learning_rate=float(learning_rate),
                    l2=float(l2),
                    epochs=int(epochs),
                ).fit(train_x[train_fit_mask], train_labels[train_fit_mask])

                validation_scores = classifier.predict_proba(validation_x[validation_mask])
                train_test_scores = classifier.predict_proba(train_x[train_test_mask])
                target_scores = classifier.predict_proba(target_x)

                threshold = None
                validation_predictions = None
                train_test_predictions = None
                target_predictions = None
                validation_positive_rate = None
                train_test_positive_rate = None
                target_positive_rate = None
                if malicious_prior is not None:
                    threshold = threshold_from_prior(validation_scores, malicious_prior)
                    validation_predictions = predict_from_threshold(validation_scores, threshold)
                    train_test_predictions = predict_from_threshold(train_test_scores, threshold)
                    target_predictions = predict_from_threshold(target_scores, threshold)
                    validation_positive_rate = float(np.mean(validation_predictions))
                    train_test_positive_rate = float(np.mean(train_test_predictions))
                    target_positive_rate = float(np.mean(target_predictions))

                row = {
                    "feature_set": triple["name"],
                    "train_name": train_name,
                    "validation_name": validation_name,
                    "target_name": target_name,
                    "train_embeddings": triple["train_embeddings"],
                    "validation_embeddings": triple["validation_embeddings"],
                    "target_embeddings": triple["target_embeddings"],
                    "train_model_id": triple["train_model_id"],
                    "validation_model_id": triple["validation_model_id"],
                    "target_model_id": triple["target_model_id"],
                    "train_layer": triple["train_layer"],
                    "validation_layer": triple["validation_layer"],
                    "target_layer": triple["target_layer"],
                    "train_pooling": triple["train_pooling"],
                    "validation_pooling": triple["validation_pooling"],
                    "target_pooling": triple["target_pooling"],
                    "feature_dim": int(train_x.shape[1]),
                    "learning_rate": float(learning_rate),
                    "l2": float(l2),
                    "epochs": int(epochs),
                    "n_train_fit_samples": int(np.sum(train_fit_mask)),
                    "n_train_test_samples": int(np.sum(train_test_mask)),
                    "n_validation_samples": int(np.sum(validation_mask)),
                    "n_target_samples": int(len(target_x)),
                    "threshold": threshold,
                    "validation_positive_rate": validation_positive_rate,
                    "train_test_positive_rate": train_test_positive_rate,
                    "target_positive_rate": target_positive_rate,
                }
                row.update(
                    {
                        f"validation_{key}": value
                        for key, value in detection_report(
                            validation_labels[validation_mask],
                            validation_scores,
                            validation_predictions,
                        ).items()
                    }
                )
                row.update(
                    {
                        f"train_test_{key}": value
                        for key, value in detection_report(
                            train_labels[train_test_mask],
                            train_test_scores,
                            train_test_predictions,
                        ).items()
                    }
                )
                row.update(
                    {
                        f"target_{key}": value
                        for key, value in detection_report(
                            target_labels,
                            target_scores,
                            target_predictions,
                        ).items()
                    }
                )
                rows.append(row)
    return rows


def _load_feature_sets(
    embedding_paths: list[Path],
    metadata: pd.DataFrame,
    id_column: str,
) -> list[dict]:
    feature_sets = []
    for embedding_path in sorted(embedding_paths):
        embeddings, sample_ids = load_embeddings(embedding_path)
        aligned_embeddings, aligned_metadata, _ = align_embeddings_with_metadata(
            embeddings,
            sample_ids,
            metadata,
            id_column=id_column,
        )
        if aligned_metadata is None:
            raise ValueError("Metadata alignment failed.")
        file_metadata = _read_embedding_file_metadata(embedding_path)
        feature_sets.append(
            {
                "name": file_metadata["pooling"] or embedding_path.stem,
                "embedding_file": str(embedding_path),
                "features": aligned_embeddings,
                "metadata": aligned_metadata,
                **file_metadata,
            }
        )
    return feature_sets


def _make_feature_triples(train_sets: list[dict], validation_sets: list[dict], target_sets: list[dict]) -> list[dict]:
    train_by_pooling = _unique_by_pooling(train_sets, "train")
    validation_by_pooling = _unique_by_pooling(validation_sets, "validation")
    target_by_pooling = _unique_by_pooling(target_sets, "target")
    common_poolings = sorted(set(train_by_pooling) & set(validation_by_pooling) & set(target_by_pooling))
    if not common_poolings:
        raise ValueError("No common pooling values found across train, validation, and target embeddings.")
    return [
        _make_feature_triple(
            name=pooling,
            train=train_by_pooling[pooling],
            validation=validation_by_pooling[pooling],
            target=target_by_pooling[pooling],
        )
        for pooling in common_poolings
    ]


def _concat_feature_triple(train_sets: list[dict], validation_sets: list[dict], target_sets: list[dict]) -> dict:
    train_by_pooling = _unique_by_pooling(train_sets, "train")
    validation_by_pooling = _unique_by_pooling(validation_sets, "validation")
    target_by_pooling = _unique_by_pooling(target_sets, "target")
    common_poolings = sorted(set(train_by_pooling) & set(validation_by_pooling) & set(target_by_pooling))
    if len(common_poolings) < 2:
        raise ValueError("At least two common pooling values are required for concat_all.")

    train_features = np.concatenate([train_by_pooling[pooling]["features"] for pooling in common_poolings], axis=1)
    validation_features = np.concatenate(
        [validation_by_pooling[pooling]["features"] for pooling in common_poolings],
        axis=1,
    )
    target_features = np.concatenate([target_by_pooling[pooling]["features"] for pooling in common_poolings], axis=1)

    return {
        "name": "concat_all",
        "train_embeddings": ";".join(train_by_pooling[pooling]["embedding_file"] for pooling in common_poolings),
        "validation_embeddings": ";".join(
            validation_by_pooling[pooling]["embedding_file"] for pooling in common_poolings
        ),
        "target_embeddings": ";".join(target_by_pooling[pooling]["embedding_file"] for pooling in common_poolings),
        "train_features": train_features,
        "validation_features": validation_features,
        "target_features": target_features,
        "train_metadata": train_sets[0]["metadata"],
        "validation_metadata": validation_sets[0]["metadata"],
        "target_metadata": target_sets[0]["metadata"],
        "train_model_id": "mixed",
        "validation_model_id": "mixed",
        "target_model_id": "mixed",
        "train_layer": None,
        "validation_layer": None,
        "target_layer": None,
        "train_pooling": "concat_all",
        "validation_pooling": "concat_all",
        "target_pooling": "concat_all",
    }


def _make_feature_triple(name: str, train: dict, validation: dict, target: dict) -> dict:
    feature_dim = train["features"].shape[1]
    if validation["features"].shape[1] != feature_dim or target["features"].shape[1] != feature_dim:
        raise ValueError(f"Feature dimension mismatch for pooling {name!r}.")
    return {
        "name": name,
        "train_embeddings": train["embedding_file"],
        "validation_embeddings": validation["embedding_file"],
        "target_embeddings": target["embedding_file"],
        "train_features": train["features"],
        "validation_features": validation["features"],
        "target_features": target["features"],
        "train_metadata": train["metadata"],
        "validation_metadata": validation["metadata"],
        "target_metadata": target["metadata"],
        "train_model_id": train["model_id"],
        "validation_model_id": validation["model_id"],
        "target_model_id": target["model_id"],
        "train_layer": train["layer"],
        "validation_layer": validation["layer"],
        "target_layer": target["layer"],
        "train_pooling": train["pooling"],
        "validation_pooling": validation["pooling"],
        "target_pooling": target["pooling"],
    }


def _unique_by_pooling(feature_sets: list[dict], name: str) -> dict[str, dict]:
    output = {}
    for feature_set in feature_sets:
        pooling = feature_set["pooling"]
        if pooling is None:
            raise ValueError(f"{name} embedding {feature_set['embedding_file']} lacks pooling metadata.")
        if pooling in output:
            raise ValueError(f"{name} embeddings contain duplicate pooling value {pooling!r}.")
        output[str(pooling)] = feature_set
    return output


def _find_selected_triple(feature_triples: list[dict], feature_set: str) -> dict:
    for triple in feature_triples:
        if triple["name"] == feature_set:
            return triple
    raise ValueError(f"Could not find selected feature triple {feature_set!r}.")


def _write_selected_target_scores(
    output_path: Path,
    selected: dict,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_fit_mask: np.ndarray,
    target_features: np.ndarray,
    target_metadata: pd.DataFrame,
) -> str:
    classifier = LogisticRegressionNumpy(
        learning_rate=float(selected["learning_rate"]),
        l2=float(selected["l2"]),
        epochs=int(selected["epochs"]),
    ).fit(train_features[train_fit_mask], train_labels[train_fit_mask])

    probabilities = classifier.predict_proba(target_features)
    table = target_metadata.copy()
    table["cross_source_malicious_probability"] = probabilities
    threshold = selected.get("threshold")
    if threshold is not None and not pd.isna(threshold):
        threshold = float(threshold)
        table["cross_source_threshold"] = threshold
        table["predicted_malicious"] = (probabilities >= threshold).astype(int)
    table["cross_source_train"] = selected["train_name"]
    table["cross_source_validation"] = selected["validation_name"]
    table["cross_source_learning_rate"] = float(selected["learning_rate"])
    table["cross_source_l2"] = float(selected["l2"])
    table["cross_source_epochs"] = int(selected["epochs"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return str(output_path)


def _read_embedding_file_metadata(path: Path) -> dict[str, str | int | None]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "model_id": _first_string(data, "model_id"),
            "layer": _first_int(data, "layer"),
            "pooling": _first_string(data, "pooling"),
        }


def _first_string(data, key: str) -> str | None:
    if key not in data:
        return None
    values = data[key]
    if len(values) == 0:
        return None
    return str(values[0])


def _first_int(data, key: str) -> int | None:
    if key not in data:
        return None
    values = data[key]
    if len(values) == 0:
        return None
    return int(values[0])


def _require_metadata(metadata: pd.DataFrame | None, name: str) -> pd.DataFrame:
    if metadata is None:
        raise ValueError(f"{name} metadata is required.")
    return metadata


def _require_column(metadata: pd.DataFrame, column: str, name: str) -> None:
    if column not in metadata.columns:
        raise ValueError(f"{name} is missing required column {column!r}.")


if __name__ == "__main__":
    main()
