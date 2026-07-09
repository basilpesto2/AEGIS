from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.datasets.jailbreakv import JailBreakVImportConfig, import_jailbreakv_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize local JailBreakV-28K CSV files into AEGIS metadata.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--mini-only", action="store_true")
    parser.add_argument("--image-root-prefix", default="JailBreakV_28K")
    args = parser.parse_args()

    table = import_jailbreakv_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        config=JailBreakVImportConfig(
            source_split=args.source_split,
            image_root_prefix=args.image_root_prefix,
            mini_only=args.mini_only,
        ),
    )

    print(f"Wrote {len(table)} rows to {args.output_csv}")
    print(table["label"].value_counts().to_string())
    print(table.groupby(["source_subset", "attack_style", "harm_category"]).size().to_string())


if __name__ == "__main__":
    main()
