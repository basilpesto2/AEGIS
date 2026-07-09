from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.subset import assign_grouped_splits, assign_stratified_splits, parse_split_fractions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign reproducible stratified split labels to AEGIS metadata."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--group-column", default="label")
    parser.add_argument(
        "--unit-column",
        default=None,
        help="If set, keep all rows with the same unit value in the same split.",
    )
    parser.add_argument("--split-column", default="experiment_split")
    parser.add_argument("--split-fractions", default="fit=0.5,val=0.25,test=0.25")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    table = pd.read_csv(args.input_csv)
    split_fractions = parse_split_fractions(args.split_fractions)
    if args.unit_column is None:
        output = assign_stratified_splits(
            table,
            group_column=args.group_column,
            split_fractions=split_fractions,
            split_column=args.split_column,
            seed=args.seed,
        )
    else:
        output = assign_grouped_splits(
            table,
            unit_column=args.unit_column,
            split_fractions=split_fractions,
            split_column=args.split_column,
            seed=args.seed,
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    counts = (
        output.groupby([args.split_column, args.group_column])
        .size()
        .rename("count")
        .reset_index()
    )
    summary = {
        "output": str(output_path),
        "rows": int(len(output)),
        "split_column": args.split_column,
        "unit_column": args.unit_column,
        "counts": counts.to_dict(orient="records"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
