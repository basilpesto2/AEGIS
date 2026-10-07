#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PIPELINE_ROOT.parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(REPOSITORY))

from aegis_research.bordair_dual_publish import (  # noqa: E402
    _candidate_runtime_preflight,
    publish_pair_package,
)
from aegis_research.bordair_dual_promotion import (  # noqa: E402
    validate_dual_promotion_candidate,
    validate_dual_source_candidate,
)
from aegis_research.bordair_paths import REPOSITORY, RunPaths  # noqa: E402


def publish_qualified_dual_pair(
    *, target: str, run_paths: RunPaths, destination: Path, runtime_config: Path
) -> dict[str, Any]:
    """Publish a qualified pair while keeping corpus and pair manifests distinct."""

    training_dir = run_paths.training_directory
    corpus_manifest = run_paths.manifest
    _, pair, _ = validate_dual_source_candidate(
        target=target,
        run_paths=run_paths,
        training_dir=training_dir,
        manifest_path=corpus_manifest,
    )
    destination = destination.expanduser().resolve()
    models_root = (REPOSITORY / "models" / "aegis").resolve()
    if destination.parent != models_root or destination == models_root:
        raise ValueError(
            "destination must be one named directory directly under models/aegis"
        )
    _candidate_runtime_preflight(runtime_config, pair, destination)

    def validate(pair_manifest: Path, image: Path, text: Path) -> dict[str, Any]:
        return validate_dual_promotion_candidate(
            target=target,
            run_paths=run_paths,
            evaluation_dir=run_paths.evaluation_directory,
            training_dir=training_dir,
            manifest_path=corpus_manifest,
            final_metadata_path=run_paths.final_metadata,
            text_led_metadata_path=run_paths.text_led_metadata,
            external_benign_metadata_path=run_paths.external_benign_metadata,
            cache_root=run_paths.feature_cache_directory / target / "evaluation",
            promoted_pair_manifest=pair_manifest,
            promoted_image_artifact=image,
            promoted_text_artifact=text,
            runtime_config=runtime_config,
        )

    return publish_pair_package(
        pair=pair,
        destination=destination,
        expected_target=target,
        post_validate=validate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish an already-qualified dual detector pair atomically.")
    parser.add_argument("target", choices=("qwen25vl3b", "llava05b"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    args = parser.parse_args()
    result = publish_qualified_dual_pair(
        target=args.target,
        run_paths=RunPaths(args.run_root),
        destination=args.destination,
        runtime_config=args.runtime_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
