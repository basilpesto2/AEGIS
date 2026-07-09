from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select transfer settings by worst-case source-test and cross-source validation AUPRC."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--validation-metric-column", default="validation_auprc")
    parser.add_argument("--source-test-metric-column", default="train_test_auprc")
    parser.add_argument("--selection-name", default="stability_min_auprc")
    args = parser.parse_args()

    table = pd.read_csv(args.input_csv)
    missing = [
        column
        for column in [args.validation_metric_column, args.source_test_metric_column]
        if column not in table.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output = table.copy()
    output["stability_score"] = output[
        [args.validation_metric_column, args.source_test_metric_column]
    ].min(axis=1)
    output["selection_name"] = args.selection_name
    selected = output.sort_values("stability_score", ascending=False).head(1)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)

    display_columns = [
        column
        for column in [
            "selection_name",
            "feature_set",
            args.validation_metric_column,
            args.source_test_metric_column,
            "stability_score",
            "target_auroc",
            "target_auprc",
            "target_f1",
            "target_positive_rate",
        ]
        if column in selected.columns
    ]
    print(selected[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
