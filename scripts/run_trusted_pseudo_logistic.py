from __future__ import annotations

import argparse
import glob
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
from AEGIS.self_training import (
    repeat_trusted_rows,
    sample_per_class,
    select_balanced_pseudo_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine a few trusted labels with balanced confident pseudo-labels."
    )
    parser.add_argument("--embedding-glob", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--labels-per-class", type=int, nargs="+", required=True)
    parser.add_argument("--pseudo-multipliers", type=int, nargs="+", default=[0, 2, 5, 10])
    parser.add_argument("--trusted-repeats", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.01])
    parser.add_argument("--l2-values", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--epochs-values", type=int, nargs="+", default=[300])
    parser.add_argument("--malicious-prior", type=float, default=None)
    parser.add_argument("--selection-metric", default="auprc")
    parser.add_argument("--include-concat", action="store_true")
    parser.add_argument("--label-seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="experiment_split")
    parser.add_argument("--fit-split", default="fit")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--test-split", default="test")
    args = parser.parse_args()

    feature_sets = _load_feature_sets(
        embedding_glob=args.embedding_glob,
        metadata_path=args.metadata,
        id_column=args.id_column,
        include_concat=args.include_concat,
    )

    rows: list[dict] = []
    for label_seed in args.label_seeds:
        for item in feature_sets:
            metadata = item["metadata"]
            _require_column(metadata, args.label_column)
            _require_column(metadata, args.split_column)
            labels = coerce_binary_labels(metadata[args.label_column])
            splits = metadata[args.split_column].astype(str).to_numpy()
            fit_mask = splits == args.fit_split
            validation_mask = splits == args.validation_split
            test_mask = splits == args.test_split

            for budget in args.labels_per_class:
                trusted_mask = sample_per_class(
                    labels,
                    fit_mask,
                    samples_per_class=budget,
                    seed=label_seed,
                )
                rows.extend(
                    _run_settings(
                        features=item["features"],
                        labels=labels,
                        fit_mask=fit_mask,
                        validation_mask=validation_mask,
                        test_mask=test_mask,
                        trusted_mask=trusted_mask,
                        pseudo_multipliers=args.pseudo_multipliers,
                        trusted_repeats=args.trusted_repeats,
                        learning_rates=args.learning_rates,
                        l2_values=args.l2_values,
                        epochs_values=args.epochs_values,
                        malicious_prior=args.malicious_prior,
                        label_seed=label_seed,
                        budget=budget,
                        feature_metadata=item,
                    )
                )

    metric_column = f"validation_{args.selection_metric}"
    if not rows or metric_column not in rows[0]:
        raise ValueError(f"Unknown selection metric {args.selection_metric!r}.")

    selected_by_seed_and_budget = {}
    for label_seed in sorted({row["label_seed"] for row in rows}):
        selected_by_seed_and_budget[str(label_seed)] = {}
        for budget in sorted({row["labels_per_class"] for row in rows}):
            candidates = [
                row
                for row in rows
                if row["label_seed"] == label_seed and row["labels_per_class"] == budget
            ]
            selected_by_seed_and_budget[str(label_seed)][str(budget)] = max(
                candidates,
                key=lambda row: row[metric_column],
            )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "n_feature_sets": len(feature_sets),
                "n_rows": len(rows),
                "selection_metric": args.selection_metric,
                "selected_by_seed_and_budget": selected_by_seed_and_budget,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_settings(
    features: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    trusted_mask: np.ndarray,
    pseudo_multipliers: list[int],
    trusted_repeats: list[int],
    learning_rates: list[float],
    l2_values: list[float],
    epochs_values: list[int],
    malicious_prior: float | None,
    label_seed: int,
    budget: int,
    feature_metadata: dict,
) -> list[dict]:
    trusted_indices = np.where(trusted_mask)[0]
    trusted_labels = labels[trusted_indices]
    unlabeled_indices = np.where(fit_mask & ~trusted_mask)[0]
    rows = []

    for learning_rate in learning_rates:
        for l2 in l2_values:
            for epochs in epochs_values:
                seed_classifier = LogisticRegressionNumpy(
                    learning_rate=learning_rate,
                    l2=l2,
                    epochs=epochs,
                    random_seed=label_seed,
                ).fit(features[trusted_indices], trusted_labels)
                unlabeled_probabilities = seed_classifier.predict_proba(
                    features[unlabeled_indices]
                )

                for multiplier in pseudo_multipliers:
                    pseudo_per_class = min(
                        budget * multiplier,
                        len(unlabeled_indices) // 2,
                    )
                    pseudo_indices, pseudo_labels = select_balanced_pseudo_labels(
                        unlabeled_probabilities,
                        unlabeled_indices,
                        samples_per_class=pseudo_per_class,
                    )
                    pseudo_accuracy = (
                        None
                        if len(pseudo_indices) == 0
                        else float(np.mean(labels[pseudo_indices] == pseudo_labels))
                    )

                    for trusted_repeat in trusted_repeats:
                        repeated_indices, repeated_labels = repeat_trusted_rows(
                            trusted_indices,
                            trusted_labels,
                            repeat=trusted_repeat,
                        )
                        train_indices = np.concatenate([repeated_indices, pseudo_indices])
                        train_labels = np.concatenate([repeated_labels, pseudo_labels])
                        classifier = LogisticRegressionNumpy(
                            learning_rate=learning_rate,
                            l2=l2,
                            epochs=epochs,
                            random_seed=label_seed,
                        ).fit(features[train_indices], train_labels)

                        validation_scores = classifier.predict_proba(features[validation_mask])
                        test_scores = classifier.predict_proba(features[test_mask])
                        threshold = None
                        validation_predictions = None
                        test_predictions = None
                        if malicious_prior is not None:
                            threshold = threshold_from_prior(validation_scores, malicious_prior)
                            validation_predictions = predict_from_threshold(
                                validation_scores,
                                threshold,
                            )
                            test_predictions = predict_from_threshold(test_scores, threshold)

                        validation_metrics = detection_report(
                            labels[validation_mask],
                            validation_scores,
                            validation_predictions,
                        )
                        test_metrics = detection_report(
                            labels[test_mask],
                            test_scores,
                            test_predictions,
                        )
                        row = {
                            "feature_set": feature_metadata["name"],
                            "embedding_file": feature_metadata["embedding_file"],
                            "model_id": feature_metadata["model_id"],
                            "layer": feature_metadata["layer"],
                            "pooling": feature_metadata["pooling"],
                            "feature_dim": int(features.shape[1]),
                            "labels_per_class": int(budget),
                            "label_seed": int(label_seed),
                            "pseudo_multiplier": int(multiplier),
                            "pseudo_per_class": int(pseudo_per_class),
                            "trusted_repeat": int(trusted_repeat),
                            "n_trusted": int(len(trusted_indices)),
                            "n_pseudo": int(len(pseudo_indices)),
                            "pseudo_label_accuracy": pseudo_accuracy,
                            "learning_rate": float(learning_rate),
                            "l2": float(l2),
                            "epochs": int(epochs),
                            "threshold": threshold,
                            "validation_positive_rate": (
                                None
                                if validation_predictions is None
                                else float(np.mean(validation_predictions))
                            ),
                            "test_positive_rate": (
                                None
                                if test_predictions is None
                                else float(np.mean(test_predictions))
                            ),
                        }
                        row.update(
                            {f"validation_{key}": value for key, value in validation_metrics.items()}
                        )
                        row.update({f"test_{key}": value for key, value in test_metrics.items()})
                        rows.append(row)
    return rows


def _load_feature_sets(
    embedding_glob: str,
    metadata_path: str,
    id_column: str,
    include_concat: bool,
) -> list[dict]:
    embedding_paths = sorted(Path(path) for path in glob.glob(embedding_glob))
    if not embedding_paths:
        raise ValueError(f"No embedding files matched {embedding_glob!r}.")
    metadata = load_metadata(metadata_path)
    if metadata is None:
        raise ValueError("Metadata is required.")

    feature_sets = []
    for embedding_path in embedding_paths:
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
                "name": embedding_path.stem,
                "embedding_file": str(embedding_path),
                "features": aligned_embeddings,
                "metadata": aligned_metadata,
                **file_metadata,
            }
        )

    if include_concat:
        feature_sets.append(
            {
                "name": "concat_all",
                "embedding_file": ";".join(item["embedding_file"] for item in feature_sets),
                "features": np.concatenate([item["features"] for item in feature_sets], axis=1),
                "metadata": feature_sets[0]["metadata"],
                "model_id": "mixed",
                "layer": None,
                "pooling": "concat_all",
            }
        )
    return feature_sets


def _read_embedding_file_metadata(path: Path) -> dict[str, str | int | None]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "model_id": _first_string(data, "model_id"),
            "layer": _first_int(data, "layer"),
            "pooling": _first_string(data, "pooling"),
        }


def _first_string(data, key: str) -> str | None:
    if key not in data or len(data[key]) == 0:
        return None
    return str(data[key][0])


def _first_int(data, key: str) -> int | None:
    if key not in data or len(data[key]) == 0:
        return None
    return int(data[key][0])


def _require_column(table: pd.DataFrame, column: str) -> None:
    if column not in table.columns:
        raise ValueError(f"Metadata is missing required column {column!r}.")


if __name__ == "__main__":
    main()
