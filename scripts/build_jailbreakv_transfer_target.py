from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a JailBreakV evaluation panel into a source-disjoint transfer target."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--malicious-role", default="jailbreakv_malicious")
    parser.add_argument("--benign-roles", nargs="+", required=True)
    parser.add_argument("--role-column", default="benchmark_role")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()

    table = pd.read_csv(args.input_csv)
    _require_columns(table, [args.role_column, args.label_column, "sample_id"])

    keep_roles = {args.malicious_role, *args.benign_roles}
    output = table[table[args.role_column].isin(keep_roles)].copy()
    if output.empty:
        raise ValueError(f"No rows matched roles: {sorted(keep_roles)}")

    malicious = output[output[args.role_column].eq(args.malicious_role)]
    benign = output[output[args.role_column].isin(args.benign_roles)]
    if malicious.empty:
        raise ValueError(f"No malicious rows matched role {args.malicious_role!r}.")
    if benign.empty:
        raise ValueError(f"No benign rows matched roles {args.benign_roles!r}.")
    if not malicious[args.label_column].eq("malicious").all():
        raise ValueError("The malicious role contains non-malicious labels.")
    if not benign[args.label_column].eq("benign").all():
        raise ValueError("The benign roles contain non-benign labels.")
    if output["sample_id"].duplicated().any():
        duplicated = output.loc[output["sample_id"].duplicated(), "sample_id"].head().tolist()
        raise ValueError(f"Duplicate sample IDs in output: {duplicated}")

    output["transfer_target_view"] = "+".join(sorted(keep_roles))
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Wrote {len(output)} rows to {args.output_csv}")
    print(output[args.label_column].value_counts().to_string())
    print(output.groupby([args.role_column, args.label_column]).size().to_string())


def _require_columns(table: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


if __name__ == "__main__":
    main()
