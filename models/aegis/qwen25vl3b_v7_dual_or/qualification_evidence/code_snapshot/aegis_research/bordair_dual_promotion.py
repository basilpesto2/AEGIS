from __future__ import annotations

"""Promotion-eligibility validation for dual artifact pairs.

This module never copies or promotes artifacts.  It proves that training,
independent evaluation, optional destination artifacts, and an optional runtime
configuration all bind to the same canonical pair.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from . import bordair_evaluation as base
from .bordair_dual import (
    load_detector_pair,
    runtime_detector_identity_sha256,
    sha256_file,
)
from .bordair_dual_evaluation import validate_dual_training_summary
from .bordair_paths import RunPaths


def validate_dual_source_candidate(
    *,
    target: str,
    run_paths: RunPaths,
    training_dir: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Validate the immutable training and one-shot qualification source closure."""

    base.validate_tracked_v7_corpus(run_paths)
    training_summary_path = training_dir / target / "dual_or" / "training_summary.json"
    summary, pair = validate_dual_training_summary(
        target=target,
        summary_path=training_summary_path,
        run_paths=run_paths,
        manifest_path=manifest_path,
    )
    qualification = pair.manifest.get("qualification_evidence")
    if not isinstance(qualification, dict) or qualification.get("passed") is not True:
        raise ValueError(
            "dual promotion requires the fully bound, independently validated "
            "one-shot qualification evidence"
        )
    if (
        qualification.get("qualification_subject_sha256") is None
        or qualification.get("runtime_detector_identity_sha256")
        != pair.manifest["runtime_detector_identity_sha256"]
        or qualification.get("rows") != 50
        or qualification.get("unique_sample_ids") != 50
    ):
        raise ValueError("one-shot qualification identity/count binding differs")
    return summary, pair, qualification


def validate_runtime_detector_config(
    *,
    config_path: Path,
    pair: Any,
) -> dict[str, Any]:
    """Validate with the live loader and the same target-profile path semantics."""

    from AEGIS.deployment_config import load_configured_detector, load_deployment_config
    from AEGIS.detector_set import OrDetector
    from AEGIS.target_assets import resolve_target_asset
    from AEGIS.target_profiles import get_target_profile

    resolved_config = config_path.expanduser().resolve()
    raw = base.read_json(resolved_config)
    if raw.get("schema_version") != 2:
        raise ValueError("runtime dual config must use deployment schema 2")
    if "base_dir" not in raw:
        raise ValueError("runtime dual config must declare base_dir explicitly")
    policy = raw.get("policy")
    if not isinstance(policy, dict) or any(
        name not in policy or policy[name] is not None
        for name in ("block_threshold", "review_threshold", "review_margin")
    ):
        raise ValueError(
            "runtime dual config requires explicit null global block/review thresholds"
        )
    config = load_deployment_config(resolved_config)
    if config.detector_path is not None or len(config.detector_heads) != 2:
        raise ValueError("live runtime config did not load exactly two detector heads")
    if config.target_profile != pair.manifest["target"]:
        raise ValueError("runtime target profile differs from detector pair target")
    if any(
        value is not None
        for value in (
            config.policy.block_threshold,
            config.policy.review_threshold,
            config.policy.review_margin,
        )
    ):
        raise ValueError("live runtime config has a non-null global detector threshold")

    loaded = load_configured_detector(config)
    if not isinstance(loaded, OrDetector):
        raise ValueError("live runtime loader did not construct an OR detector")
    if (
        loaded.pooling != "text_image_tokens"
        or loaded.base_feature_dim != pair.base_feature_dim
        or loaded.feature_dim != pair.base_feature_dim * 2
    ):
        raise ValueError("runtime OR detector fused pooling/dimension differs from pair")
    profile = get_target_profile(config.target_profile)
    if profile.name != pair.manifest["target"] or profile.detector_mode != "or":
        raise ValueError("runtime target profile is not the pair's dual target")
    if config.provider_spec != profile.provider or config.provider_options != profile.provider_options:
        raise ValueError("runtime provider configuration differs from target profile")
    shared = pair.manifest["shared_provenance"]
    expected_provider = {
        "model_id": shared["model_id"],
        "model_revision": shared["model_revision"],
        "tokenizer_revision": shared["tokenizer_revision"],
        "layer": shared["layer"],
        "pooling": "text_image_tokens",
        "feature_dim": pair.base_feature_dim * 2,
    }
    if profile.model_family != shared["model_family"] or any(
        config.provider_options.get(name) != value
        for name, value in expected_provider.items()
    ):
        raise ValueError("runtime provider pooling/dimension/provenance differs from pair")

    configured_heads = {head.name: head for head in config.detector_heads}
    profile_heads = {head.name: head for head in profile.detector_heads}
    loaded_heads = {head.name: head for head in loaded.heads}
    if set(configured_heads) != {"image", "text"} or set(profile_heads) != set(
        configured_heads
    ):
        raise ValueError("runtime/profile detector head sets differ")
    identity_entries: dict[str, dict[str, Any]] = {}
    head_evidence: dict[str, dict[str, Any]] = {}
    for name in ("image", "text"):
        configured = configured_heads[name]
        expected_profile = profile_heads[name]
        runtime_head = loaded_heads[name]
        pair_entry = pair.manifest["artifacts"][name]
        profile_path = resolve_target_asset(
            expected_profile.detector, source_root=config.base_dir
        ).path
        if configured.artifact_path != profile_path:
            raise ValueError(f"runtime {name} path differs from target profile")
        digest = sha256_file(configured.artifact_path)
        if digest != pair_entry["sha256"] or digest != expected_profile.detector_sha256:
            raise ValueError(f"runtime {name} artifact bytes differ from pair/profile")
        if float(configured.review_threshold) != float(pair_entry["review_threshold"]) or float(
            configured.review_threshold
        ) != float(expected_profile.review_threshold):
            raise ValueError(f"runtime {name} review threshold differs from pair/profile")
        if (
            runtime_head.artifact_sha256 != digest
            or runtime_head.artifact.pooling != pair_entry["pooling"]
            or runtime_head.artifact.feature_dim != pair_entry["feature_dim"]
            or float(runtime_head.artifact.threshold)
            != float(pair_entry["block_threshold"])
            or runtime_head.artifact.source != pair_entry["source"]
        ):
            raise ValueError(f"runtime {name} artifact contract differs from pair")
        for field in (
            "model_family",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "preprocessing_sha256",
            "layer",
        ):
            if getattr(runtime_head.artifact, field) != shared[field]:
                raise ValueError(f"runtime {name} provenance differs for {field}")
        identity_entries[name] = {
            "sha256": digest,
            "review_threshold": float(configured.review_threshold),
        }
        head_evidence[name] = {
            "path": base.display_path(configured.artifact_path),
            "sha256": digest,
            "pooling": runtime_head.artifact.pooling,
            "feature_dim": runtime_head.artifact.feature_dim,
            "block_threshold": float(runtime_head.artifact.threshold),
            "review_threshold": float(configured.review_threshold),
        }
    runtime_identity = runtime_detector_identity_sha256(identity_entries)
    if (
        runtime_identity != pair.manifest["runtime_detector_identity_sha256"]
        or loaded.identity_sha256 != runtime_identity
        or profile.detector_identity_sha256 != runtime_identity
    ):
        raise ValueError("runtime, profile, and pair detector identities differ")
    return {
        "schema_version": 2,
        "path": base.display_path(resolved_config),
        "sha256": sha256_file(resolved_config),
        "base_dir": base.display_path(config.base_dir),
        "target_profile": config.target_profile,
        "pooling": loaded.pooling,
        "base_feature_dim": loaded.base_feature_dim,
        "feature_dim": loaded.feature_dim,
        "provenance": {
            name: shared[name]
            for name in (
                "model_family",
                "model_id",
                "model_revision",
                "tokenizer_revision",
                "preprocessing_sha256",
                "layer",
            )
        },
        "runtime_detector_identity_sha256": runtime_identity,
        "heads": head_evidence,
        "global_thresholds_are_null": True,
    }


def validate_dual_promotion_candidate(
    *,
    target: str,
    run_paths: RunPaths,
    evaluation_dir: Path,
    training_dir: Path,
    manifest_path: Path,
    final_metadata_path: Path,
    text_led_metadata_path: Path,
    external_benign_metadata_path: Path,
    cache_root: Path,
    promoted_pair_manifest: Path | None = None,
    promoted_image_artifact: Path | None = None,
    promoted_text_artifact: Path | None = None,
    runtime_config: Path | None = None,
) -> dict[str, Any]:
    training_summary_path = training_dir / target / "dual_or" / "training_summary.json"
    summary, pair, qualification = validate_dual_source_candidate(
        target=target,
        run_paths=run_paths,
        training_dir=training_dir,
        manifest_path=manifest_path,
    )
    stable_identity = pair.manifest["pair_identity_sha256"]

    destination: dict[str, Any] | None = None
    if any(
        value is not None
        for value in (
            promoted_pair_manifest,
            promoted_image_artifact,
            promoted_text_artifact,
        )
    ):
        if not all(
            value is not None
            for value in (
                promoted_pair_manifest,
                promoted_image_artifact,
                promoted_text_artifact,
            )
        ):
            raise ValueError("destination validation requires manifest and both artifacts")
        assert promoted_pair_manifest is not None
        assert promoted_image_artifact is not None
        assert promoted_text_artifact is not None
        if sha256_file(promoted_pair_manifest) != sha256_file(pair.manifest_path):
            raise ValueError("destination pair manifest is not byte-identical")
        destination_pair = load_detector_pair(
            promoted_pair_manifest,
            expected_target=target,
            expected_base_feature_dim=pair.base_feature_dim,
            image_override=promoted_image_artifact,
            text_override=promoted_text_artifact,
        )
        if destination_pair.manifest["pair_identity_sha256"] != stable_identity:
            raise ValueError("destination pair identity differs from candidate")
        destination = {
            "pair_manifest": base.display_path(promoted_pair_manifest),
            "pair_manifest_sha256": sha256_file(promoted_pair_manifest),
            "image_artifact_sha256": sha256_file(promoted_image_artifact),
            "text_artifact_sha256": sha256_file(promoted_text_artifact),
        }

    if runtime_config is None:
        raise ValueError(
            "dual promotion requires a schema-2 runtime config validated by the live loader"
        )
    config_evidence = validate_runtime_detector_config(
        config_path=runtime_config, pair=pair
    )
    return {
        "schema_version": 1,
        "validation_kind": "dual_or_promotion_eligibility",
        "target": target,
        "passed": True,
        "pair_identity_sha256": stable_identity,
        "runtime_detector_identity_sha256": pair.manifest[
            "runtime_detector_identity_sha256"
        ],
        "training_summary_sha256": sha256_file(training_summary_path),
        "pair_manifest_sha256": sha256_file(pair.manifest_path),
        "artifacts": pair.manifest["artifacts"],
        "training_acceptance": summary["acceptance"],
        "one_shot_qualification_evidence": qualification,
        "frozen_panels_rerun": False,
        "destination": destination,
        "runtime_config": config_evidence,
        "mutations_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dual-head detector promotion eligibility without copying files."
    )
    parser.add_argument("target", choices=tuple(base.EXPECTED_TARGETS))
    parser.add_argument("--run-root", type=Path, default=base.RUN)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--final-metadata", type=Path)
    parser.add_argument("--text-led-metadata", type=Path)
    parser.add_argument("--external-benign-metadata", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--promoted-pair-manifest", type=Path)
    parser.add_argument("--promoted-image-artifact", type=Path)
    parser.add_argument("--promoted-text-artifact", type=Path)
    parser.add_argument("--runtime-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = RunPaths(args.run_root)
    evaluation_dir = paths.choose(args.evaluation_dir, paths.evaluation_directory)
    training_dir = paths.choose(args.training_dir, paths.training_directory)
    manifest_path = paths.corpus_input(args.manifest, paths.manifest, "--manifest")
    result = validate_dual_promotion_candidate(
        target=args.target,
        run_paths=paths,
        evaluation_dir=evaluation_dir,
        training_dir=training_dir,
        manifest_path=manifest_path,
        final_metadata_path=paths.corpus_input(
            args.final_metadata, paths.final_metadata, "--final-metadata"
        ),
        text_led_metadata_path=paths.corpus_input(
            args.text_led_metadata, paths.text_led_metadata, "--text-led-metadata"
        ),
        external_benign_metadata_path=paths.corpus_input(
            args.external_benign_metadata,
            paths.external_benign_metadata,
            "--external-benign-metadata",
        ),
        cache_root=(
            paths.feature_cache_directory / args.target / "evaluation"
            if args.cache_root is None
            else args.cache_root.expanduser().resolve()
        ),
        promoted_pair_manifest=args.promoted_pair_manifest,
        promoted_image_artifact=args.promoted_image_artifact,
        promoted_text_artifact=args.promoted_text_artifact,
        runtime_config=args.runtime_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
