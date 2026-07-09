from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.evidence import build_evidence_report, render_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a requirement-by-requirement production evidence report."
    )
    parser.add_argument("--json-output", default="docs/production_evidence_report.json")
    parser.add_argument("--markdown-output", default="docs/production_evidence_report.md")
    args = parser.parse_args()

    report = build_evidence_report(ROOT)
    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_output.write_text(render_markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "json_output": str(json_output),
                "markdown_output": str(markdown_output),
                "status_counts": report["status_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
