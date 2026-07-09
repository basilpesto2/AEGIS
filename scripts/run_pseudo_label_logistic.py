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

from AEGIS.evaluation import run_pseudo_label_logistic_validation_grid
from AEGIS.io import align_embeddings_with_metadata, load_embeddings, load_metadata
from AEGIS.labels import coerce_binary_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train logistic classifiers from SVD pseudo-labels and select on validation."
    )
    parser.add_argument("--embedding-glob", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--svd-components", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--orientations", nargs="+", default=["normal", "inverted"])
    parser.add_argument("--benign-keep-fractions", nargs="+", default=["all", "0.167"])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    parser.add_argument("--l2-values", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--epochs-values", type=int, nargs="+", default=[300, 600])
    parser.add_argument("--malicious-prior", type=float, required=True)
    parser.add_argument("--selection-metric", default="auprc")
    parser.add_argument("--random-seed", type=int, default=42)
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
        raise ValueError("Metadata is required for pseudo-label training.")
    benign_keep_fractions = _parse_optional_floats(args.benign_keep_fractions)

    rows: list[dict] = []
    for embedding_path in embedding_paths:
        embeddings, sample_ids = load_embeddings(embedding_path)
        aligned_embeddings, aligned_metadata, _ = align_embeddings_with_metadata(
            embeddings,
            sample_ids,
            metadata,
            id_column=args.id_column,
        )
        if aligned_metadata is None:
            raise ValueError("Metadata alignment failed.")
        if args.label_column not in aligned_metadata.columns:
            raise ValueError(f"Metadata is missing label column {args.label_column!r}.")
        if args.split_column not in aligned_metadata.columns:
            raise ValueError(f"Metadata is missing split column {args.split_column!r}.")

        labels = coerce_binary_labels(aligned_metadata[args.label_column])
        split_values = aligned_metadata[args.split_column].astype(str).to_numpy()
        results = run_pseudo_label_logistic_validation_grid(
            aligned_embeddings,
            labels=labels,
            fit_mask=split_values == args.fit_split,
            validation_mask=split_values == args.validation_split,
            test_mask=split_values == args.test_split,
            svd_components=args.svd_components,
            orientations=args.orientations,
            malicious_prior=args.malicious_prior,
            benign_keep_fractions=benign_keep_fractions,
            learning_rates=args.learning_rates,
            l2_values=args.l2_values,
            epochs_values=args.epochs_values,
            random_seed=args.random_seed,
        )
        file_metadata = _read_embedding_file_metadata(embedding_path)
        for result in results:
            row = result.as_row()
            row.update(file_metadata)
            row["embedding_file"] = str(embedding_path)
            row["feature_set"] = embedding_path.stem
            row["benign_keep_fraction_label"] = (
                "all" if row["benign_keep_fraction"] is None else row["benign_keep_fraction"]
            )
            rows.append(row)

    metric_column = f"validation_{args.selection_metric}"
    if not rows or metric_column not in rows[0]:
        available = sorted(key.removeprefix("validation_") for key in rows[0] if key.startswith("validation_"))
        raise ValueError(
            f"Unknown selection metric {args.selection_metric!r}. Available metrics: {available}"
        )

    selected = max(rows, key=lambda row: row[metric_column])
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)

    print(
        json.dumps(
            {
                "output": str(output),
                "n_embedding_files": len(embedding_paths),
                "n_rows": len(rows),
                "selection_metric": args.selection_metric,
                "selected": selected,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _parse_optional_floats(values: list[str]) -> list[float | None]:
    parsed: list[float | None] = []
    for value in values:
        if value.strip().lower() in {"all", "none"}:
            parsed.append(None)
        else:
            parsed.append(float(value))
    return parsed


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


if __name__ == "__main__":
    main()
