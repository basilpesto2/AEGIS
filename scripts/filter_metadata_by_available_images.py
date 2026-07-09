from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only metadata rows whose image files exist locally.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--image-column", default="image_path")
    args = parser.parse_args()

    table = pd.read_csv(args.input_csv)
    if args.image_column not in table.columns:
        raise ValueError(f"Missing image column: {args.image_column}")

    corpus_root = Path(args.corpus_root)
    exists = table[args.image_column].fillna("").map(lambda value: bool(value) and (corpus_root / str(value)).exists())
    kept = table[exists].copy()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output_path, index=False)

    print(f"Wrote {len(kept)} of {len(table)} rows to {args.output_csv}")
    print(f"Missing or unavailable images: {len(table) - len(kept)}")
    if "label" in kept.columns:
        print(kept["label"].value_counts().to_string())
    grouping = [column for column in ["source_subset", "attack_style", "harm_category"] if column in kept.columns]
    if grouping:
        print(kept.groupby(grouping).size().to_string())


if __name__ == "__main__":
    main()
