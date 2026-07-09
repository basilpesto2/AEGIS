from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.production_gate import (
    ProductionGateCriteria,
    build_production_gate_report,
)
from AEGIS.readiness import doctor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether supplied AEGIS evidence satisfies a deployment "
            "readiness gate. Passing this gate validates only the supplied scope."
        )
    )
    parser.add_argument(
        "--summary-json",
        nargs="+",
        required=True,
        help="Evaluation summary JSON files, such as litmus or benchmark summaries.",
    )
    parser.add_argument("--detector", default="models/aegis/aegis_qwen25vl3b_text_detector.npz")
    parser.add_argument("--manifest", default="docs/reproducibility_manifest.json")
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--scope", default="controlled_deployment")
    parser.add_argument("--min-samples", type=int, default=32)
    parser.add_argument("--min-summary-files", type=int, default=1)
    parser.add_argument("--min-model-families", type=int, default=1)
    parser.add_argument("--min-sources", type=int, default=1)
    parser.add_argument("--required-model-family", action="append", default=[])
    parser.add_argument("--required-source", action="append", default=[])
    parser.add_argument("--min-auroc", type=float, default=0.95)
    parser.add_argument("--min-auprc", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--max-fnr", type=float, default=0.10)
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    criteria = ProductionGateCriteria(
        min_samples=args.min_samples,
        min_summary_files=args.min_summary_files,
        min_model_families=args.min_model_families,
        min_sources=args.min_sources,
        required_model_families=tuple(args.required_model_family),
        required_sources=tuple(args.required_source),
        min_auroc=args.min_auroc,
        min_auprc=args.min_auprc,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        max_false_positive_rate=args.max_fpr,
        max_false_negative_rate=args.max_fnr,
        require_clean_git=args.require_clean_git,
    )
    report = build_production_gate_report(
        summary_paths=[Path(path) for path in args.summary_json],
        criteria=criteria,
        manifest_path=args.manifest,
        doctor_report=doctor(args.detector, cache_dir=args.cache_dir),
        git_dirty=_git_dirty(),
        scope=args.scope,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not bool(report["ok"]):
        raise SystemExit(1)


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


if __name__ == "__main__":
    main()
