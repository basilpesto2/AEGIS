from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.evaluation import run_svd_component_grid
from AEGIS.io import align_embeddings_with_metadata, load_embeddings, load_metadata
from AEGIS.labels import coerce_binary_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SVD detector ablations over components and score orientation."
    )
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--components", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--orientations", nargs="+", default=["normal", "inverted"])
    parser.add_argument("--malicious-prior", type=float, default=None)
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--fit-split", default=None)
    parser.add_argument("--eval-split", default=None)
    args = parser.parse_args()

    embeddings, sample_ids = load_embeddings(args.embeddings)
    metadata = load_metadata(args.metadata)
    embeddings, metadata, _ = align_embeddings_with_metadata(
        embeddings,
        sample_ids,
        metadata,
        id_column=args.id_column,
    )
    if metadata is None:
        raise ValueError("Metadata is required for ablation.")

    labels = None
    if args.label_column in metadata.columns:
        labels = coerce_binary_labels(metadata[args.label_column])

    split_values = None
    if args.fit_split is not None or args.eval_split is not None:
        if args.split_column not in metadata.columns:
            raise ValueError(f"Metadata is missing split column {args.split_column!r}.")
        split_values = metadata[args.split_column].astype(str).to_numpy()

    fit_mask = None
    if args.fit_split is not None:
        fit_mask = split_values == args.fit_split

    eval_mask = None
    if args.eval_split is not None:
        eval_mask = split_values == args.eval_split

    results = run_svd_component_grid(
        embeddings,
        components=args.components,
        labels=labels,
        fit_mask=fit_mask,
        eval_mask=eval_mask,
        orientations=args.orientations,
        malicious_prior=args.malicious_prior,
    )
    rows = [result.as_row() for result in results]

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)

    summary = {
        "output": str(output),
        "n_rows": len(rows),
        "best_by_auroc": _best_row(rows, "auroc"),
        "best_by_auprc": _best_row(rows, "auprc"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _best_row(rows: list[dict], metric: str) -> dict | None:
    metric_rows = [row for row in rows if row.get(metric) is not None]
    if not metric_rows:
        return None
    return max(metric_rows, key=lambda row: row[metric])


if __name__ == "__main__":
    main()
