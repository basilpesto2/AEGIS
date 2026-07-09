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
from AEGIS.holdout import make_family_holdout_masks, make_paired_family_holdout_masks
from AEGIS.io import align_embeddings_with_metadata, load_embeddings, load_metadata
from AEGIS.labels import coerce_binary_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate logistic detectors with malicious attack families held out for test."
    )
    parser.add_argument("--embeddings", required=True, nargs="+")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--family-column", default="source_subset")
    parser.add_argument("--pair-column", default=None)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--seeds", type=int, nargs="+", default=[81, 82, 83])
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    parser.add_argument("--l2-values", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--epochs-values", type=int, nargs="+", default=[300, 600])
    parser.add_argument("--malicious-prior", type=float, default=0.5)
    parser.add_argument("--selection-metric", default="auprc")
    parser.add_argument("--include-concat", action="store_true")
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    if metadata is None:
        raise ValueError("Metadata is required.")
    feature_sets = _load_feature_sets(
        [Path(path) for path in args.embeddings],
        metadata,
        id_column=args.id_column,
        include_concat=args.include_concat,
    )

    aligned_metadata = feature_sets[0]["metadata"]
    _require_column(aligned_metadata, args.label_column)
    _require_column(aligned_metadata, args.family_column)
    if args.pair_column is not None:
        _require_column(aligned_metadata, args.pair_column)
    labels = coerce_binary_labels(aligned_metadata[args.label_column])
    families = aligned_metadata[args.family_column].astype(str).to_numpy()
    pair_ids = (
        None
        if args.pair_column is None
        else aligned_metadata[args.pair_column].astype(str).to_numpy()
    )

    rows: list[dict] = []
    for heldout_family in args.families:
        for seed in args.seeds:
            if pair_ids is None:
                fit_mask, validation_mask, test_mask = make_family_holdout_masks(
                    labels,
                    families,
                    heldout_family=heldout_family,
                    validation_fraction=args.validation_fraction,
                    seed=seed,
                )
            else:
                fit_mask, validation_mask, test_mask = make_paired_family_holdout_masks(
                    labels,
                    families,
                    pair_ids,
                    heldout_family=heldout_family,
                    validation_fraction=args.validation_fraction,
                    seed=seed,
                )
            for item in feature_sets:
                results = run_logistic_validation_grid(
                    item["features"],
                    labels=labels,
                    fit_mask=fit_mask,
                    validation_mask=validation_mask,
                    test_mask=test_mask,
                    learning_rates=args.learning_rates,
                    l2_values=args.l2_values,
                    epochs_values=args.epochs_values,
                    malicious_prior=args.malicious_prior,
                    random_seed=seed,
                )
                for result in results:
                    row = result.as_row()
                    row.update(
                        {
                            "heldout_family": heldout_family,
                            "holdout_seed": seed,
                            "feature_set": item["name"],
                            "embedding_file": item["embedding_file"],
                            "model_id": item["model_id"],
                            "layer": item["layer"],
                            "pooling": item["pooling"],
                            "feature_dim": int(item["features"].shape[1]),
                            "n_heldout_malicious": int(
                                np.sum(test_mask & (labels == 1))
                            ),
                        }
                    )
                    rows.append(row)

    metric_column = f"validation_{args.selection_metric}"
    selected = []
    for heldout_family in args.families:
        for seed in args.seeds:
            candidates = [
                row
                for row in rows
                if row["heldout_family"] == heldout_family and row["holdout_seed"] == seed
            ]
            selected.append(max(candidates, key=lambda row: row[metric_column]))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    selected_path = output_path.with_name(f"{output_path.stem}_selected.csv")
    pd.DataFrame(selected).to_csv(selected_path, index=False)

    summary = (
        pd.DataFrame(selected)
        .groupby("heldout_family")
        .agg(
            test_auroc_mean=("test_auroc", "mean"),
            test_auroc_std=("test_auroc", "std"),
            test_auprc_mean=("test_auprc", "mean"),
            test_auprc_std=("test_auprc", "std"),
            test_f1_mean=("test_f1", "mean"),
            test_f1_std=("test_f1", "std"),
            n=("test_auprc", "size"),
        )
        .reset_index()
    )
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "selected_output": str(selected_path),
                "summary_output": str(summary_path),
                "n_rows": len(rows),
                "selected": selected,
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _load_feature_sets(
    embedding_paths: list[Path],
    metadata: pd.DataFrame,
    id_column: str,
    include_concat: bool,
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
