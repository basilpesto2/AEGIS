from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.deployment_package import build_target_deployment_package


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a portable deployment package from an AEGIS target bundle."
    )
    parser.add_argument("--bundle", required=True, help="Validated target bundle JSON path.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the package.")
    parser.add_argument("--root", default=".", help="Root used to resolve source bundle paths.")
    parser.add_argument("--require-labeled-monitoring", action="store_true")
    parser.add_argument("--require-robustness-evidence", action="store_true")
    parser.add_argument("--require-attack-success-evidence", action="store_true")
    parser.add_argument(
        "--required-provider-modality",
        action="append",
        default=[],
        choices=["text", "image", "image_text"],
        help="Additional provider modality coverage to require; repeatable.",
    )
    args = parser.parse_args()

    report = build_target_deployment_package(
        args.bundle,
        args.output_dir,
        root=args.root,
        require_labeled_monitoring=args.require_labeled_monitoring,
        require_robustness_evidence=args.require_robustness_evidence,
        require_attack_success_evidence=args.require_attack_success_evidence,
        required_provider_modalities=tuple(args.required_provider_modality),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not bool(report.get("ok")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
