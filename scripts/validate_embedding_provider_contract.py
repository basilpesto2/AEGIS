from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.detector_artifact import load_detector_artifact
from AEGIS.guardrail import GuardrailRequest
from AEGIS.provider_contract import load_provider, validate_provider_contract


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that an embedding provider satisfies the AEGIS runtime contract."
    )
    parser.add_argument("--provider", required=True, help="Provider spec: module:object.")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", default="Summarize the visible content safely.")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional image path for providers that require image-text requests; repeatable.",
    )
    parser.add_argument("--request-id", default="contract-request")
    parser.add_argument(
        "--required-modality",
        action="append",
        default=[],
        choices=["text", "image", "image_text"],
        help="Require the contract sample set to exercise this request modality; repeatable.",
    )
    parser.add_argument("--allow-provenance-mismatch", action="store_true")
    parser.add_argument("--allow-nondeterministic", action="store_true")
    args = parser.parse_args()

    provider = load_provider(args.provider)
    artifact = load_detector_artifact(args.detector)
    report = validate_provider_contract(
        provider,
        artifact,
        requests=[
            GuardrailRequest(
                text=args.text,
                image_paths=tuple(args.image),
                request_id=args.request_id,
            )
        ],
        require_matching_provenance=not args.allow_provenance_mismatch,
        require_deterministic=not args.allow_nondeterministic,
        required_modalities=tuple(args.required_modality),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not bool(report["ok"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
