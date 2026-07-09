from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.subset import parse_group_counts, stratified_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reproducible stratified AEGIS metadata subset."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--group-column", default="label")
    parser.add_argument("--n-per-group", type=int, default=None)
    parser.add_argument(
        "--group-counts",
        default=None,
        help="Comma-separated counts, e.g. benign=100,malicious=20.",
    )
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    table = pd.read_csv(args.input_csv)
    group_counts = parse_group_counts(args.group_counts) if args.group_counts else None
    subset = stratified_sample(
        table,
        group_column=args.group_column,
        n_per_group=args.n_per_group,
        group_counts=group_counts,
        groups=args.groups,
        seed=args.seed,
        strict=args.strict,
        shuffle=not args.no_shuffle,
    )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output, index=False)

    summary = {
        "output": str(output),
        "rows": int(len(subset)),
        "group_column": args.group_column,
        "counts": {
            str(key): int(value)
            for key, value in subset[args.group_column].astype(str).value_counts().sort_index().items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
