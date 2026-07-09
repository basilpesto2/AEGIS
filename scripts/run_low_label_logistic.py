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

from AEGIS.evaluation import run_logistic_validation_grid
from AEGIS.experiment_io import load_feature_sets
from AEGIS.io import load_metadata
from AEGIS.labels import coerce_binary_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run label-budgeted supervised logistic baselines."
    )
    parser.add_argument("--embedding-glob", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--labels-per-class", type=int, nargs="+", required=True)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    parser.add_argument("--l2-values", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--epochs-values", type=int, nargs="+", default=[300, 600])
    parser.add_argument("--malicious-prior", type=float, default=None)
    parser.add_argument("--selection-metric", default="auprc")
    parser.add_argument("--include-concat", action="store_true")
    parser.add_argument("--label-seed", type=int, default=101)
    parser.add_argument(
        "--label-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of label-sampling seeds. Overrides --label-seed when set.",
    )
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="experiment_split")
    parser.add_argument("--fit-split", default="fit")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--test-split", default="test")
    args = parser.parse_args()

    embedding_paths = sorted(Path(path) for path in glob.glob(args.embedding_glob))
    if not embedding_paths:
        raise ValueError(f"No embedding files matched {args.embedding_glob!r}.")

    metadata = load_metadata(args.metadata)
    if metadata is None:
        raise ValueError("Metadata is required for low-label training.")

    feature_sets = load_feature_sets(embedding_paths, metadata, id_column=args.id_column)
    for feature_set in feature_sets:
        feature_set["name"] = Path(feature_set["embedding_file"]).stem

    if args.include_concat:
        base_metadata = feature_sets[0]["metadata"]
        concatenated = np.concatenate([item["features"] for item in feature_sets], axis=1)
        feature_sets.append(
            {
                "name": "concat_all",
                "embedding_file": ";".join(item["embedding_file"] for item in feature_sets),
                "features": concatenated,
                "metadata": base_metadata,
                "model_id": "mixed",
                "layer": None,
                "pooling": "concat_all",
            }
        )

    label_seeds = args.label_seeds if args.label_seeds is not None else [args.label_seed]

    rows: list[dict] = []
    for label_seed in label_seeds:
        for item in feature_sets:
            aligned_embeddings = item["features"]
            aligned_metadata = item["metadata"]
            if args.label_column not in aligned_metadata.columns:
                raise ValueError(f"Metadata is missing label column {args.label_column!r}.")
            if args.split_column not in aligned_metadata.columns:
                raise ValueError(f"Metadata is missing split column {args.split_column!r}.")

            labels = coerce_binary_labels(aligned_metadata[args.label_column])
            split_values = aligned_metadata[args.split_column].astype(str).to_numpy()
            full_fit_mask = split_values == args.fit_split
            validation_mask = split_values == args.validation_split
            test_mask = split_values == args.test_split

            for budget in args.labels_per_class:
                low_label_fit_mask = _sample_per_class(
                    labels,
                    full_fit_mask,
                    labels_per_class=budget,
                    seed=label_seed,
                )
                results = run_logistic_validation_grid(
                    aligned_embeddings,
                    labels=labels,
                    fit_mask=low_label_fit_mask,
                    validation_mask=validation_mask,
                    test_mask=test_mask,
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
                            "model_id": item["model_id"],
                            "layer": item["layer"],
                            "pooling": item["pooling"],
                        }
                    )
                    row["embedding_file"] = item["embedding_file"]
                    row["feature_set"] = item["name"]
                    row["feature_dim"] = int(aligned_embeddings.shape[1])
                    row["labels_per_class"] = int(budget)
                    row["label_seed"] = int(label_seed)
                    row["n_labeled_fit"] = int(np.sum(low_label_fit_mask))
                    rows.append(row)

    metric_column = f"validation_{args.selection_metric}"
    if not rows or metric_column not in rows[0]:
        available = sorted(key.removeprefix("validation_") for key in rows[0] if key.startswith("validation_"))
        raise ValueError(
            f"Unknown selection metric {args.selection_metric!r}. Available metrics: {available}"
        )

    selected_by_budget = {}
    for budget in sorted({row["labels_per_class"] for row in rows}):
        budget_rows = [row for row in rows if row["labels_per_class"] == budget]
        selected_by_budget[str(budget)] = max(budget_rows, key=lambda row: row[metric_column])

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
                "n_feature_sets": len(feature_sets),
                "label_seeds": label_seeds,
                "n_rows": len(rows),
                "selection_metric": args.selection_metric,
                "selected_by_budget": selected_by_budget,
                "selected_by_seed_and_budget": selected_by_seed_and_budget,
            },
            indent=2,
            sort_keys=True,
        )
    )


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


if __name__ == "__main__":
    main()
