from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.datasets.mssbench import MSSBenchImportConfig, import_mssbench_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize local MSSBench combined.json into AEGIS metadata.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--exclude-chat", action="store_true")
    parser.add_argument("--exclude-embodied", action="store_true")
    args = parser.parse_args()

    table = import_mssbench_json(
        input_json=args.input_json,
        output_csv=args.output_csv,
        config=MSSBenchImportConfig(
            source_split=args.source_split,
            include_chat=not args.exclude_chat,
            include_embodied=not args.exclude_embodied,
        ),
    )

    print(f"Wrote {len(table)} rows to {args.output_csv}")
    print(table["label"].value_counts().to_string())
    print(table.groupby(["task_type", "source_subset", "label"]).size().to_string())


if __name__ == "__main__":
    main()
