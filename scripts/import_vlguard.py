from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.datasets.vlguard import VLGuardImportConfig, import_vlguard_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize local VLGuard JSON into AEGIS metadata.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--source-split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--exclude-safe-safes", action="store_true")
    parser.add_argument("--exclude-safe-unsafes", action="store_true")
    parser.add_argument("--exclude-unsafes", action="store_true")
    args = parser.parse_args()

    table = import_vlguard_json(
        input_json=args.input_json,
        output_csv=args.output_csv,
        config=VLGuardImportConfig(
            source_split=args.source_split,
            include_safe_safes=not args.exclude_safe_safes,
            include_safe_unsafes=not args.exclude_safe_unsafes,
            include_unsafes=not args.exclude_unsafes,
        ),
    )

    print(f"Wrote {len(table)} rows to {args.output_csv}")
    print(table["label"].value_counts().to_string())
    print(table.groupby(["source_subset", "label"]).size().to_string())


if __name__ == "__main__":
    main()

