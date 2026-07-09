from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one validation-best low-label row per seed/budget and summarize test metrics."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--selected-output-csv", required=True)
    parser.add_argument("--summary-output-csv", required=True)
    parser.add_argument("--selection-metric", default="auprc")
    args = parser.parse_args()

    table = pd.read_csv(args.input_csv)
    required = {"label_seed", "labels_per_class", f"validation_{args.selection_metric}"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    selection_column = f"validation_{args.selection_metric}"
    selected_indices = (
        table.groupby(["label_seed", "labels_per_class"], sort=True)[selection_column]
        .idxmax()
        .to_numpy()
    )
    selected = table.loc[selected_indices].sort_values(
        ["labels_per_class", "label_seed"]
    )
    metric_columns = [
        column
        for column in ["test_auroc", "test_auprc", "test_fpr95", "test_f1"]
        if column in selected.columns
    ]
    summary = selected.groupby("labels_per_class")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["n_seeds"] = selected.groupby("labels_per_class").size().to_numpy()

    selected_path = Path(args.selected_output_csv)
    summary_path = Path(args.summary_output_csv)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selected_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(
        json.dumps(
            {
                "selected_output": str(selected_path),
                "summary_output": str(summary_path),
                "selected_rows": int(len(selected)),
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
