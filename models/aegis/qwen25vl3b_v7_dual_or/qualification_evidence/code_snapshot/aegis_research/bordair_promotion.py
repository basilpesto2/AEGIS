from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[3]
RUN = REPOSITORY / "outputs" / "bordair_retraining_v1"
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from . import bordair_evaluation as evaluation  # noqa: E402
from . import bordair_evaluation_validation as evaluation_validator  # noqa: E402
from AEGIS.detector_artifact import load_detector_artifact  # noqa: E402
from AEGIS.target_profiles import get_target_profile  # noqa: E402


TARGETS = {
    "llava05b": "configs/aegis.llava.deployment.container.json",
    "qwen25vl3b": "configs/aegis.deployment.container.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the promoted v7 artifacts against target profiles, deployment "
            "configs, training summaries, and complete runtime evaluation evidence."
        )
    )
    parser.add_argument("targets", nargs="*", choices=tuple(TARGETS), default=list(TARGETS))
    parser.add_argument(
        "--artifact-mode",
        choices=("fused", "dual_or"),
        default="fused",
        help="Validate historical fused promotion or canonical dual-pair promotion.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=evaluation.RUN,
        help=(
            "Root containing the generated v7 corpus, training outputs, and "
            "evaluation evidence. Every unspecified path is derived from it."
        ),
    )
    parser.add_argument(
        "--evaluation-dir", type=Path
    )
    parser.add_argument(
        "--training-dir", type=Path
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--final-metadata", type=Path
    )
    parser.add_argument(
        "--text-led-metadata", type=Path
    )
    parser.add_argument(
        "--external-benign-metadata",
        type=Path,
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--promoted-pair-manifest", type=Path)
    parser.add_argument("--promoted-image-artifact", type=Path)
    parser.add_argument("--promoted-text-artifact", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    return parser.parse_args()


def resolve_cli_paths(args: argparse.Namespace) -> tuple[
    evaluation.RunPaths, Path, Path, Path, Path, Path, Path
]:
    paths = evaluation.RunPaths(args.run_root)
    return (
        paths,
        paths.choose(args.evaluation_dir, paths.evaluation_directory),
        paths.choose(args.training_dir, paths.training_directory),
        paths.corpus_input(args.manifest, paths.manifest, "--manifest"),
        paths.corpus_input(
            args.final_metadata, paths.final_metadata, "--final-metadata"
        ),
        paths.corpus_input(
            args.text_led_metadata, paths.text_led_metadata, "--text-led-metadata"
        ),
        paths.corpus_input(
            args.external_benign_metadata,
            paths.external_benign_metadata,
            "--external-benign-metadata",
        ),
    )


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object in {path}.")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_profile_config_artifact(
    target: str,
) -> tuple[object, Path, str, float, Path]:
    profile = get_target_profile(target)
    config_path = (REPOSITORY / TARGETS[target]).resolve()
    config = read_json(config_path)
    require(config.get("target_profile") == target, f"Config target mismatch for {target}.")
    require(config.get("detector") == profile.detector, f"Config detector mismatch for {target}.")
    require(config.get("provider") == profile.provider, f"Config provider mismatch for {target}.")
    require(
        config.get("provider_options") == profile.provider_options,
        f"Config/provider-option mismatch for {target}.",
    )
    policy = config.get("policy")
    require(isinstance(policy, dict), f"Config policy is missing for {target}.")
    assert isinstance(policy, dict)
    require(
        policy.get("block_threshold") is None,
        f"Config overrides block threshold for {target}.",
    )
    require(
        evaluation.close_float(policy.get("review_threshold"), profile.review_threshold),
        f"Config/profile review-threshold mismatch for {target}.",
    )

    artifact_path = (REPOSITORY / profile.detector).resolve()
    require(artifact_path.is_file(), f"Promoted detector is missing for {target}: {artifact_path}")
    artifact_hash = evaluation.sha256_file(artifact_path)
    require(
        artifact_hash == profile.detector_sha256,
        f"Target-profile detector hash mismatch for {target}.",
    )
    artifact = load_detector_artifact(artifact_path)
    options = profile.provider_options
    require(artifact.model_family == profile.model_family, f"Model-family mismatch for {target}.")
    for name in ("model_id", "model_revision", "tokenizer_revision", "layer", "pooling"):
        require(
            getattr(artifact, name) == options.get(name),
            f"Artifact/profile {name} mismatch for {target}: "
            f"{getattr(artifact, name)!r} != {options.get(name)!r}",
        )
    require(
        artifact.feature_dim == options.get("feature_dim"),
        f"Artifact/profile feature_dim mismatch for {target}.",
    )
    base_dim = int(evaluation.EXPECTED_TARGETS[target]["base_feature_dim"])
    require(
        artifact.pooling == "text_image_tokens",
        f"Promoted v7 detector must use text_image_tokens for {target}.",
    )
    require(
        artifact.feature_dim == base_dim * 2,
        f"Promoted v7 detector must use the doubled fused dimension for {target}.",
    )
    require(
        profile.review_threshold is not None
        and 0.0 <= float(profile.review_threshold) < artifact.threshold < 1.0,
        f"Invalid promoted thresholds for {target}.",
    )
    return artifact, artifact_path, artifact_hash, float(profile.review_threshold), config_path


def validate_target(
    target: str,
    args: argparse.Namespace,
    *,
    run_paths: evaluation.RunPaths | None = None,
    corpus_validated: bool = False,
) -> dict[str, object]:
    paths = run_paths or evaluation.RunPaths(args.run_root)
    if not corpus_validated:
        evaluation.validate_tracked_v7_corpus(paths)
    artifact, artifact_path, artifact_hash, review_threshold, config_path = (
        validate_profile_config_artifact(target)
    )
    training_dir = paths.choose(args.training_dir, paths.training_directory)
    manifest_path = paths.corpus_input(args.manifest, paths.manifest, "--manifest")
    evaluation_dir = paths.choose(args.evaluation_dir, paths.evaluation_directory)
    final_metadata_path = paths.corpus_input(
        args.final_metadata, paths.final_metadata, "--final-metadata"
    )
    text_led_metadata_path = paths.corpus_input(
        args.text_led_metadata, paths.text_led_metadata, "--text-led-metadata"
    )
    external_benign_metadata_path = paths.corpus_input(
        args.external_benign_metadata,
        paths.external_benign_metadata,
        "--external-benign-metadata",
    )
    summary_path = training_dir / target / "training_summary.json"
    summary, training_artifact, _, training_hash, training_review = (
        evaluation.validate_artifact_and_summary(
            target,
            summary_path,
            artifact_path,
            manifest_path=manifest_path,
            development_metadata_path=manifest_path.parent / "development_metadata_v7.csv",
            regression_metadata_path=manifest_path.parent / "regression_metadata_v7.csv",
            run_root=paths.root,
        )
    )
    require(training_hash == artifact_hash, f"Training/promoted artifact mismatch for {target}.")
    require(
        training_artifact.pooling == artifact.pooling
        and training_artifact.feature_dim == artifact.feature_dim,
        f"Training/promoted representation mismatch for {target}.",
    )
    require(
        evaluation.close_float(training_review, review_threshold),
        f"Training/profile review-threshold mismatch for {target}.",
    )
    require(
        summary.get("artifact_sha256") == artifact_hash,
        f"Training-summary artifact hash mismatch for {target}.",
    )
    selected = summary.get("selected_candidate")
    require(isinstance(selected, dict), f"Training summary has no selected candidate for {target}.")
    assert isinstance(selected, dict)
    require(
        selected.get("pooling") == artifact.pooling,
        f"Selected-candidate pooling mismatch for {target}.",
    )
    require(
        evaluation.close_float(selected.get("block_threshold"), artifact.threshold),
        f"Selected-candidate block threshold mismatch for {target}.",
    )
    require(
        evaluation.close_float(selected.get("review_threshold"), review_threshold),
        f"Selected-candidate review threshold mismatch for {target}.",
    )
    internal_test = evaluation.internal_test_evidence(summary)
    require(
        internal_test.get("passed") is True,
        f"Internal-test zero-FN promotion gate failed for {target}.",
    )

    runtime = evaluation_validator.validate_target_evaluation(
        target,
        evaluation_dir=evaluation_dir,
        training_dir=training_dir,
        manifest_path=manifest_path,
        final_metadata_path=final_metadata_path,
        text_led_metadata_path=text_led_metadata_path,
        external_benign_metadata_path=external_benign_metadata_path,
        run_paths=paths,
        corpus_validated=True,
    )
    require(
        runtime.get("artifact_sha256") == artifact_hash,
        f"Runtime/promoted artifact mismatch for {target}.",
    )
    acceptance = runtime.get("acceptance")
    require(isinstance(acceptance, dict), f"Runtime acceptance is missing for {target}.")
    assert isinstance(acceptance, dict)
    checks = acceptance.get("checks")
    require(isinstance(checks, dict), f"Runtime acceptance checks are missing for {target}.")
    assert isinstance(checks, dict)
    for name in (
        "regression_all_malicious_blocked",
        "regression_fewer_than_three_benign_blocked",
        "text_led_all_malicious_blocked",
        "external_benign_fewer_than_three_blocked",
        "training_internal_test_has_zero_false_negatives",
        "all_runtime_decisions_match_static_scores",
    ):
        require(checks.get(name) is True, f"Promotion check {name!r} failed for {target}.")

    return {
        "target": target,
        "config": evaluation.display_path(config_path),
        "artifact": evaluation.display_path(artifact_path),
        "artifact_sha256": artifact_hash,
        "pooling": artifact.pooling,
        "feature_dim": artifact.feature_dim,
        "block_threshold": artifact.threshold,
        "review_threshold": review_threshold,
        "internal_test": internal_test,
        "runtime_evaluation": runtime,
    }


def main() -> None:
    args = parse_args()
    (
        paths,
        evaluation_dir,
        training_dir,
        manifest_path,
        final_metadata_path,
        text_led_metadata_path,
        external_benign_metadata_path,
    ) = resolve_cli_paths(args)
    if getattr(args, "artifact_mode", "fused") == "dual_or":
        from .bordair_dual_promotion import validate_dual_promotion_candidate

        targets = list(args.targets) if args.targets else list(TARGETS)
        output = [
            validate_dual_promotion_candidate(
                target=target,
                run_paths=paths,
                evaluation_dir=evaluation_dir,
                training_dir=training_dir,
                manifest_path=manifest_path,
                final_metadata_path=final_metadata_path,
                text_led_metadata_path=text_led_metadata_path,
                external_benign_metadata_path=external_benign_metadata_path,
                cache_root=(
                    paths.feature_cache_directory / target / "evaluation"
                    if getattr(args, "cache_root", None) is None
                    else args.cache_root.expanduser().resolve()
                ),
                promoted_pair_manifest=getattr(args, "promoted_pair_manifest", None),
                promoted_image_artifact=getattr(args, "promoted_image_artifact", None),
                promoted_text_artifact=getattr(args, "promoted_text_artifact", None),
                runtime_config=getattr(args, "runtime_config", None),
            )
            for target in targets
        ]
        print(json.dumps({"ok": True, "targets": output}, indent=2, sort_keys=True))
        return
    evaluation.validate_tracked_v7_corpus(paths)
    targets = list(args.targets) if args.targets else list(TARGETS)
    output = [
        validate_target(
            target,
            args,
            run_paths=paths,
            corpus_validated=True,
        )
        for target in targets
    ]
    print(json.dumps({"ok": True, "targets": output}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
