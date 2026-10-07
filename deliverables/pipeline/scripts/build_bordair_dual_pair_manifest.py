from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PIPELINE_ROOT.parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(REPOSITORY))

from AEGIS.detector_artifact import load_detector_artifact  # noqa: E402
from aegis_research.bordair_dual import (  # noqa: E402
    build_pair_manifest,
    build_one_shot_qualification_binding,
    sha256_file,
)
from aegis_research.bordair_provenance import DETECTOR_CORPUS_VERSION  # noqa: E402
from aegis_research.bordair_provenance import training_identity_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bind two existing standard v1 channel artifacts into a canonical "
            "Bordair dual-pair manifest."
        )
    )
    parser.add_argument("target")
    parser.add_argument("--image-artifact", type=Path, required=True)
    parser.add_argument("--text-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--training-identity-sha256")
    identity.add_argument(
        "--development-summary",
        type=Path,
        help=(
            "Development-only experiment summary whose declared_grid, protocol, "
            "and scalar selected_channel_pair fields define the training identity."
        ),
    )
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--development-metadata", type=Path, required=True)
    parser.add_argument("--regression-metadata", type=Path, required=True)
    parser.add_argument("--image-review-threshold", type=float)
    parser.add_argument("--text-review-threshold", type=float)
    parser.add_argument("--qualification-evidence-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    image_path = args.image_artifact.expanduser().resolve()
    text_path = args.text_artifact.expanduser().resolve()
    image = load_detector_artifact(image_path)
    text = load_detector_artifact(text_path)
    training_identity = args.training_identity_sha256
    if args.development_summary is not None:
        development_summary = json.loads(
            args.development_summary.read_text(encoding="utf-8")
        )
        if not isinstance(development_summary, dict):
            raise ValueError("development summary must contain an object")
        selected = development_summary.get("selected_channel_pair")
        if not isinstance(selected, dict):
            raise ValueError("development summary has no selected_channel_pair")
        scalar_selected = {
            key: value
            for key, value in selected.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        training_identity = training_identity_sha256(
            {
                "schema_version": 1,
                "declared_grid": development_summary.get("declared_grid"),
                "protocol": development_summary.get("protocol"),
            },
            scalar_selected,
        )
    assert isinstance(training_identity, str)
    image_review = (
        max(0.0, float(image.threshold) - float(image.uncertainty_margin))
        if args.image_review_threshold is None
        else float(args.image_review_threshold)
    )
    text_review = (
        max(0.0, float(text.threshold) - float(text.uncertainty_margin))
        if args.text_review_threshold is None
        else float(args.text_review_threshold)
    )
    corpus_manifest_hash = sha256_file(args.corpus_manifest)
    qualification = (
        None
        if args.qualification_evidence_dir is None
        else build_one_shot_qualification_binding(
            evidence_directory=args.qualification_evidence_dir,
            pair_directory=output.parent,
            image_artifact_sha256=sha256_file(image_path),
            text_artifact_sha256=sha256_file(text_path),
            corpus_manifest_sha256=corpus_manifest_hash,
        )
    )
    manifest = build_pair_manifest(
        target=args.target,
        image_path=image_path,
        image_artifact=image,
        image_review_threshold=image_review,
        text_path=text_path,
        text_artifact=text,
        text_review_threshold=text_review,
        output_directory=output.parent,
        training_identity_sha256_value=training_identity,
        corpus={
            "version": DETECTOR_CORPUS_VERSION,
            "manifest_sha256": corpus_manifest_hash,
            "development_metadata_sha256": sha256_file(args.development_metadata),
            "regression_metadata_sha256": sha256_file(args.regression_metadata),
        },
        qualification_evidence=qualification,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
