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

from AEGIS.evaluation import run_logistic_validation_grid
from AEGIS.io import align_embeddings_with_metadata, load_embeddings, load_metadata
from AEGIS.labels import coerce_binary_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train hybrid supervised logistic detectors from source-domain labels plus "
            "a small target-domain label budget."
        )
    )
    parser.add_argument("--source-embeddings", required=True, nargs="+")
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--target-embeddings", required=True, nargs="+")
    parser.add_argument("--target-metadata", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--source-name", default="source")
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--labels-per-class", type=int, nargs="+", required=True)
    parser.add_argument("--label-seeds", type=int, nargs="+", default=[101])
    parser.add_argument(
        "--source-samples-per-class-values",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional source-domain samples per class to include in each fit set. "
            "Omit to use all source fit rows."
        ),
    )
    parser.add_argument("--target-repeat-values", type=int, nargs="+", default=[1])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    parser.add_argument("--l2-values", type=float, nargs="+", default=[1e-4, 1e-2])
    parser.add_argument("--epochs-values", type=int, nargs="+", default=[300])
    parser.add_argument("--malicious-prior", type=float, default=None)
    parser.add_argument("--selection-metric", default="auprc")
    parser.add_argument("--include-concat", action="store_true")
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="experiment_split")
    parser.add_argument("--fit-split", default="fit")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--test-split", default="test")
    args = parser.parse_args()

    source_metadata = _require_metadata(load_metadata(args.source_metadata), "source")
    target_metadata = _require_metadata(load_metadata(args.target_metadata), "target")

    source_sets = _load_feature_sets(
        [Path(path) for path in args.source_embeddings],
        source_metadata,
        id_column=args.id_column,
    )
    target_sets = _load_feature_sets(
        [Path(path) for path in args.target_embeddings],
        target_metadata,
        id_column=args.id_column,
    )
    feature_pairs = _pair_feature_sets(source_sets, target_sets)
    if args.include_concat:
        feature_pairs.append(_concat_feature_pair(source_sets, target_sets))

    source_meta = feature_pairs[0]["source_metadata"]
    target_meta = feature_pairs[0]["target_metadata"]
    _require_column(source_meta, args.label_column, "source metadata")
    _require_column(source_meta, args.split_column, "source metadata")
    _require_column(target_meta, args.label_column, "target metadata")
    _require_column(target_meta, args.split_column, "target metadata")

    source_labels = coerce_binary_labels(source_meta[args.label_column])
    target_labels = coerce_binary_labels(target_meta[args.label_column])
    source_splits = source_meta[args.split_column].astype(str).to_numpy()
    target_splits = target_meta[args.split_column].astype(str).to_numpy()
    source_fit_mask = source_splits == args.fit_split
    target_full_fit_mask = target_splits == args.fit_split
    target_validation_mask = target_splits == args.validation_split
    target_test_mask = target_splits == args.test_split
    source_samples_per_class_values = (
        args.source_samples_per_class_values
        if args.source_samples_per_class_values is not None
        else [None]
    )

    rows: list[dict] = []
    for pair in feature_pairs:
        for label_seed in args.label_seeds:
            for source_samples_per_class in source_samples_per_class_values:
                if source_samples_per_class is None:
                    source_control_mask = source_fit_mask
                else:
                    source_control_mask = _sample_per_class(
                        source_labels,
                        source_fit_mask,
                        labels_per_class=source_samples_per_class,
                        seed=label_seed + 100_000,
                    )
                for budget in args.labels_per_class:
                    target_low_label_mask = _sample_per_class(
                        target_labels,
                        target_full_fit_mask,
                        labels_per_class=budget,
                        seed=label_seed,
                    )
                    for target_repeat in args.target_repeat_values:
                        combined = _build_hybrid_arrays(
                            source_features=pair["source_features"],
                            source_labels=source_labels,
                            source_fit_mask=source_control_mask,
                            target_features=pair["target_features"],
                            target_labels=target_labels,
                            target_low_label_mask=target_low_label_mask,
                            target_validation_mask=target_validation_mask,
                            target_test_mask=target_test_mask,
                            target_repeat=target_repeat,
                        )
                        results = run_logistic_validation_grid(
                            combined["features"],
                            labels=combined["labels"],
                            fit_mask=combined["fit_mask"],
                            validation_mask=combined["validation_mask"],
                            test_mask=combined["test_mask"],
                            learning_rates=args.learning_rates,
                            l2_values=args.l2_values,
                            epochs_values=args.epochs_values,
                            malicious_prior=args.malicious_prior,
                            random_seed=label_seed,
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
                                    "labels_per_class": int(budget),
                                    "label_seed": int(label_seed),
                                    "source_samples_per_class": source_samples_per_class,
                                    "target_repeat": int(target_repeat),
                                    "n_source_fit": int(np.sum(source_control_mask)),
                                    "n_target_labeled_fit": int(np.sum(target_low_label_mask)),
                                }
                            )
                            rows.append(row)

    metric_column = f"validation_{args.selection_metric}"
    if not rows or metric_column not in rows[0]:
        available = sorted(
            key.removeprefix("validation_")
            for key in rows[0]
            if key.startswith("validation_")
        )
        raise ValueError(
            f"Unknown selection metric {args.selection_metric!r}. Available metrics: {available}"
        )

    selected_by_seed_and_budget = {}
    for label_seed in sorted({row["label_seed"] for row in rows}):
        seed_rows = [row for row in rows if row["label_seed"] == label_seed]
        selected_by_seed_and_budget[str(label_seed)] = {}
        for budget in sorted({row["labels_per_class"] for row in seed_rows}):
            budget_rows = [row for row in seed_rows if row["labels_per_class"] == budget]
            selected_by_seed_and_budget[str(label_seed)][str(budget)] = max(
                budget_rows,
                key=lambda row: row[metric_column],
            )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)

    print(
        json.dumps(
            {
                "output": str(output),
                "n_feature_sets": len(feature_pairs),
                "n_rows": len(rows),
                "label_seeds": args.label_seeds,
                "selection_metric": args.selection_metric,
                "selected_by_seed_and_budget": selected_by_seed_and_budget,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _build_hybrid_arrays(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    source_fit_mask: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    target_low_label_mask: np.ndarray,
    target_validation_mask: np.ndarray,
    target_test_mask: np.ndarray,
    target_repeat: int,
) -> dict[str, np.ndarray]:
    if target_repeat < 1:
        raise ValueError("target_repeat must be positive.")

    source_train_x = source_features[source_fit_mask]
    source_train_y = source_labels[source_fit_mask]
    target_train_x = np.repeat(target_features[target_low_label_mask], target_repeat, axis=0)
    target_train_y = np.repeat(target_labels[target_low_label_mask], target_repeat, axis=0)

    train_x = np.vstack([source_train_x, target_train_x])
    train_y = np.concatenate([source_train_y, target_train_y])
    features = np.vstack([train_x, target_features])
    labels = np.concatenate([train_y, target_labels])

    n_train = len(train_y)
    n_target = len(target_labels)
    fit_mask = np.concatenate([np.ones(n_train, dtype=bool), np.zeros(n_target, dtype=bool)])
    validation_mask = np.concatenate([np.zeros(n_train, dtype=bool), target_validation_mask])
    test_mask = np.concatenate([np.zeros(n_train, dtype=bool), target_test_mask])
    return {
        "features": features,
        "labels": labels,
        "fit_mask": fit_mask,
        "validation_mask": validation_mask,
        "test_mask": test_mask,
    }


def _sample_per_class(
    labels: np.ndarray,
    fit_mask: np.ndarray,
    labels_per_class: int,
    seed: int,
) -> np.ndarray:
    if labels_per_class < 1:
        raise ValueError("labels_per_class must be positive.")
    rng = np.random.default_rng(seed)
    output = np.zeros(len(labels), dtype=bool)
    for label in (0, 1):
        candidates = np.where(fit_mask & (labels == label))[0]
        if len(candidates) < labels_per_class:
            raise ValueError(
                f"Class {label} has only {len(candidates)} fit examples, "
                f"fewer than requested {labels_per_class}."
            )
        selected = rng.choice(candidates, size=labels_per_class, replace=False)
        output[selected] = True
    return output


def _load_aligned_features(
    embedding_path: Path,
    metadata: pd.DataFrame,
    id_column: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings, sample_ids = load_embeddings(embedding_path)
    aligned_embeddings, aligned_metadata, _ = align_embeddings_with_metadata(
        embeddings,
        sample_ids,
        metadata,
        id_column=id_column,
    )
    if aligned_metadata is None:
        raise ValueError("Metadata alignment failed.")
    return aligned_embeddings, aligned_metadata


def _load_feature_sets(
    embedding_paths: list[Path],
    metadata: pd.DataFrame,
    id_column: str,
) -> list[dict]:
    feature_sets = []
    for embedding_path in sorted(embedding_paths):
        embeddings, aligned_metadata = _load_aligned_features(
            embedding_path,
            metadata,
            id_column=id_column,
        )
        file_metadata = _read_embedding_file_metadata(embedding_path)
        feature_sets.append(
            {
                "name": file_metadata["pooling"] or embedding_path.stem,
                "embedding_file": str(embedding_path),
                "features": embeddings,
                "metadata": aligned_metadata,
                **file_metadata,
            }
        )
    return feature_sets


def _pair_feature_sets(source_sets: list[dict], target_sets: list[dict]) -> list[dict]:
    if len(source_sets) == 1 and len(target_sets) == 1:
        return [_make_feature_pair(source_sets[0], target_sets[0])]

    source_by_pooling = _unique_by_pooling(source_sets, "source")
    target_by_pooling = _unique_by_pooling(target_sets, "target")
    common_poolings = sorted(set(source_by_pooling) & set(target_by_pooling))
    if not common_poolings:
        raise ValueError("No common pooling values found between source and target embeddings.")

    return [
        _make_feature_pair(source_by_pooling[pooling], target_by_pooling[pooling])
        for pooling in common_poolings
    ]


def _concat_feature_pair(source_sets: list[dict], target_sets: list[dict]) -> dict:
    source_by_pooling = _unique_by_pooling(source_sets, "source")
    target_by_pooling = _unique_by_pooling(target_sets, "target")
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
