from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.target_release import validate_target_registry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate every production-candidate target in an AEGIS target registry."
    )
    parser.add_argument("--registry", default="docs/target_validation_registry.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--min-production-candidates", type=int, default=1)
    parser.add_argument("--min-model-families", type=int, default=1)
    parser.add_argument("--required-model-family", action="append", default=[])
    parser.add_argument(
        "--required-target-modality",
        action="append",
        default=[],
        choices=["text", "image", "image_text"],
        help="Require the production-candidate registry to include this target modality; repeatable.",
    )
    parser.add_argument("--require-labeled-monitoring", action="store_true")
    parser.add_argument("--require-robustness-evidence", action="store_true")
    parser.add_argument("--require-attack-success-evidence", action="store_true")
    parser.add_argument(
        "--required-provider-modality",
        action="append",
        default=[],
        choices=["text", "image", "image_text"],
        help="Require each provider contract report to cover this request modality; repeatable.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = validate_target_registry(
        args.registry,
        root=args.root,
        min_production_candidates=args.min_production_candidates,
        min_model_families=args.min_model_families,
        required_model_families=tuple(args.required_model_family),
        required_target_modalities=tuple(args.required_target_modality),
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
