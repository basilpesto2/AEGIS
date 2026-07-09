from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.monitoring import summarize_decisions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize guardrail decision logs and optional drift from a baseline."
    )
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()

    decisions = pd.read_csv(args.decisions)
    summary = summarize_decisions(
        decisions,
        label_column=args.label_column,
        baseline_path=args.baseline,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
