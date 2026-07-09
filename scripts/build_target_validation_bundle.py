from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.production_gate import ProductionGateCriteria, build_production_gate_report
from AEGIS.readiness import doctor
from AEGIS.target_validation import (
    build_target_validation_bundle,
    render_target_bundle_markdown,
    update_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an auditable validation bundle for one target LLM/MLLM."
    )
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--scope", default="controlled_deployment")
    parser.add_argument(
        "--intended-modality",
        action="append",
        default=[],
        choices=["text", "image", "image_text"],
        help="Request modality this target deployment intends to guard; repeatable.",
    )
    parser.add_argument("--detector", required=True)
    parser.add_argument("--summary-json", nargs="+", required=True)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--provider-contract", default=None)
    parser.add_argument("--calibration-summary", default=None)
    parser.add_argument("--monitoring-summary", default=None)
    parser.add_argument(
        "--robustness-summary",
        action="append",
        default=[],
        help="Optional robustness/adaptive evaluation artifact for this target; repeatable.",
    )
    parser.add_argument(
        "--attack-success-summary",
        action="append",
        default=[],
        help="Optional guarded response or attack-success reduction artifact; repeatable.",
    )
    parser.add_argument("--manifest", default="docs/reproducibility_manifest.json")
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--notes", default="")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--registry", default="docs/target_validation_registry.json")
    parser.add_argument("--min-samples", type=int, default=32)
    parser.add_argument("--min-auroc", type=float, default=0.95)
    parser.add_argument("--min-auprc", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--max-fnr", type=float, default=0.10)
    args = parser.parse_args()

    criteria = ProductionGateCriteria(
        min_samples=args.min_samples,
        min_auroc=args.min_auroc,
        min_auprc=args.min_auprc,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        max_false_positive_rate=args.max_fpr,
        max_false_negative_rate=args.max_fnr,
        required_model_families=(args.model_family,),
    )
    gate_report = build_production_gate_report(
        summary_paths=[Path(path) for path in args.summary_json],
        criteria=criteria,
        manifest_path=args.manifest,
        doctor_report=doctor(args.detector, cache_dir=args.cache_dir),
        scope=args.scope,
    )
    bundle = build_target_validation_bundle(
        target_name=args.target_name,
        model_family=args.model_family,
        model_id=args.model_id,
        scope=args.scope,
        detector_path=args.detector,
        summary_paths=args.summary_json,
        gate_report=gate_report,
        intended_modalities=args.intended_modality,
        policy_path=args.policy,
        provider_contract_path=args.provider_contract,
        calibration_summary_path=args.calibration_summary,
        monitoring_summary_path=args.monitoring_summary,
        manifest_path=args.manifest,
        robustness_summary_paths=args.robustness_summary,
        attack_success_summary_paths=args.attack_success_summary,
        notes=args.notes,
    )

    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    markdown_output.write_text(render_target_bundle_markdown(bundle), encoding="utf-8")
    registry = update_registry(args.registry, bundle)
    print(
        json.dumps(
            {
                "bundle_status": bundle.status,
                "json_output": str(json_output),
                "markdown_output": str(markdown_output),
                "registry": args.registry,
                "registry_status_counts": registry["status_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
