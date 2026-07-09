from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize selected transfer scores without printing prompt text.")
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=["benchmark_role", "source_subset", "label"],
    )
    parser.add_argument("--score-column", default="transfer_malicious_probability")
    parser.add_argument("--prediction-column", default="predicted_malicious")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()

    table = pd.read_csv(args.score_csv)
    missing = [
        column
        for column in [args.score_column, args.prediction_column, args.label_column, *args.group_columns]
        if column not in table.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    table["_is_malicious"] = table[args.label_column].eq("malicious")
    summary = (
        table.groupby(args.group_columns, dropna=False)
        .agg(
            n=("sample_id", "size"),
            mean_score=(args.score_column, "mean"),
            median_score=(args.score_column, "median"),
            predicted_positive_rate=(args.prediction_column, "mean"),
            malicious_rate=("_is_malicious", "mean"),
        )
        .reset_index()
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"Wrote {len(summary)} rows to {args.output_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
