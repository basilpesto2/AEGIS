from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.target_release import validate_target_release


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an AEGIS target bundle as a releasable guardrail package."
    )
    parser.add_argument("--bundle", required=True, help="Target bundle JSON path.")
    parser.add_argument("--root", default=".", help="Root used to resolve bundle artifact paths.")
    parser.add_argument("--require-status", default="production_candidate")
    parser.add_argument("--require-labeled-monitoring", action="store_true")
    parser.add_argument("--require-robustness-evidence", action="store_true")
    parser.add_argument("--require-attack-success-evidence", action="store_true")
    parser.add_argument(
        "--required-provider-modality",
        action="append",
        default=[],
        choices=["text", "image", "image_text"],
        help="Require the provider contract report to cover this request modality; repeatable.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    report = validate_target_release(
        args.bundle,
        root=args.root,
        require_status=args.require_status,
        require_labeled_monitoring=args.require_labeled_monitoring,
        require_robustness_evidence=args.require_robustness_evidence,
        require_attack_success_evidence=args.require_attack_success_evidence,
        required_provider_modalities=tuple(args.required_provider_modality),
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not bool(report["ok"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
