from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from . import bordair_evaluation as base


TARGETS = tuple(base.EXPECTED_TARGETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate v7 runtime evaluation JSON/CSV evidence, recompute every "
            "score-derived action, and bind it to the corpus, training summary, "
            "detector artifact, and all three frozen promotion panels."
        )
    )
    parser.add_argument("targets", nargs="*", choices=TARGETS, default=list(TARGETS))
    parser.add_argument(
        "--artifact-mode",
        choices=("fused", "dual_or"),
        default="fused",
        help="Validate historical fused evidence or dual-pair evidence.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=base.RUN,
        help=(
            "Root containing the generated v7 corpus, training outputs, and "
            "evaluation evidence. Every unspecified path is derived from it."
        ),
    )
    parser.add_argument(
        "--evaluation-dir", type=Path
    )
    parser.add_argument("--training-dir", type=Path)
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
    return parser.parse_args()


def resolve_cli_paths(args: argparse.Namespace) -> tuple[
    base.RunPaths, Path, Path, Path, Path, Path, Path
]:
    paths = base.RunPaths(args.run_root)
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


def resolve_reported_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a reported file path, got {value!r}.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base.REPOSITORY / path).resolve()


def _number(value: object, description: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be numeric, not boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{description} is not finite: {value!r}")
    return number


def _require_close(left: object, right: object, description: str) -> None:
    if not base.close_float(left, right):
        raise ValueError(f"{description} mismatch: {left!r} != {right!r}")


def _csv_cell(name: str, value: object) -> str:
    if name in {"reasons", "decision_image_sha256"}:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return "" if value is None else str(value)


def validate_csv_copy(csv_path: Path, rows: list[dict[str, object]]) -> None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != base.CSV_FIELDS:
            raise ValueError(f"Unexpected v7 evaluation CSV columns: {csv_path}")
        csv_rows = list(reader)
    if len(csv_rows) != len(rows):
        raise ValueError(
            f"CSV/JSON row-count mismatch: {len(csv_rows)} != {len(rows)}"
        )
    for index, (csv_row, json_row) in enumerate(zip(csv_rows, rows, strict=True), start=1):
        for name in base.CSV_FIELDS:
            expected = _csv_cell(name, json_row.get(name))
            if csv_row.get(name) != expected:
                raise ValueError(
                    f"CSV/JSON mismatch at row {index}, field {name!r}: "
                    f"{csv_row.get(name)!r} != {expected!r}"
                )


def _artifact_payload(artifact, path: Path, artifact_hash: str, review: float) -> dict[str, object]:
    return {
        "path": base.display_path(path),
        "sha256": artifact_hash,
        "source": artifact.source,
        "model_family": artifact.model_family,
        "model_id": artifact.model_id,
        "model_revision": artifact.model_revision,
        "tokenizer_revision": artifact.tokenizer_revision,
        "preprocessing_sha256": artifact.preprocessing_sha256,
        "layer": artifact.layer,
        "pooling": artifact.pooling,
        "feature_dim": artifact.feature_dim,
        "block_threshold": float(artifact.threshold),
        "review_threshold": review,
    }


def validate_fused_representation(target: str, artifact: object) -> None:
    expected = base.EXPECTED_TARGETS[target]
    expected_dimension = int(expected["base_feature_dim"]) * 2
    if getattr(artifact, "pooling", None) != "text_image_tokens":
        raise ValueError(
            f"v7 evaluation requires fused text_image_tokens pooling for {target}."
        )
    if getattr(artifact, "feature_dim", None) != expected_dimension:
        raise ValueError(
            f"v7 evaluation requires feature_dim={expected_dimension} for {target}; "
            f"got {getattr(artifact, 'feature_dim', None)!r}."
        )


def validate_target_evaluation(
    target: str,
    *,
    evaluation_dir: Path,
    training_dir: Path,
    manifest_path: Path,
    final_metadata_path: Path,
    text_led_metadata_path: Path,
    external_benign_metadata_path: Path,
    run_paths: base.RunPaths | None = None,
    corpus_validated: bool = False,
) -> dict[str, object]:
    manifest_path = manifest_path.expanduser().resolve()
    final_metadata_path = final_metadata_path.expanduser().resolve()
    text_led_metadata_path = text_led_metadata_path.expanduser().resolve()
    external_benign_metadata_path = external_benign_metadata_path.expanduser().resolve()
    evaluation_dir = evaluation_dir.expanduser().resolve()
    training_dir = training_dir.expanduser().resolve()
    paths = run_paths or base.RunPaths(manifest_path.parent)
    if not corpus_validated:
        base.validate_tracked_v7_corpus(paths)
    manifest, panels, metadata_info = base.validate_v7_manifest_and_panels(
        manifest_path,
        final_metadata_path,
        text_led_metadata_path,
        external_benign_metadata_path,
    )

    json_path = evaluation_dir / f"{target}_runtime_results.json"
    csv_path = evaluation_dir / f"{target}_runtime_results.csv"
    result = base.read_json(json_path)
    if result.get("schema_version") != 2 or result.get("evaluation_version") != "v7":
        raise ValueError(f"Unsupported v7 evaluation schema for {target}.")
    if result.get("target") != target:
        raise ValueError(f"Evaluation target mismatch for {target}.")
    if result.get("dataset") != manifest.get("dataset"):
        raise ValueError(f"Evaluation dataset mismatch for {target}.")
    if result.get("dataset_commit") != manifest.get("dataset_commit"):
        raise ValueError(f"Evaluation dataset revision mismatch for {target}.")

    manifest_report = result.get("corpus_manifest")
    expected_manifest_report = {
        "path": base.display_path(manifest_path),
        "sha256": base.sha256_file(manifest_path),
    }
    if manifest_report != expected_manifest_report:
        raise ValueError(f"Evaluation manifest binding mismatch for {target}.")
    if result.get("metadata") != metadata_info:
        raise ValueError(f"Evaluation metadata binding mismatch for {target}.")
    expected_metadata_hashes = {
        panel: str(info["sha256"]) for panel, info in metadata_info.items()
    }
    if result.get("metadata_sha256") != expected_metadata_hashes:
        raise ValueError(f"Evaluation metadata hashes mismatch for {target}.")

    summary_path = training_dir / target / "training_summary.json"
    if resolve_reported_path(result.get("training_summary_path")) != summary_path:
        raise ValueError(f"Evaluation training-summary path mismatch for {target}.")
    if result.get("training_summary_sha256") != base.sha256_file(summary_path):
        raise ValueError(f"Evaluation training-summary hash mismatch for {target}.")
    artifact_report = result.get("artifact")
    if not isinstance(artifact_report, dict):
        raise ValueError(f"Evaluation has no artifact object for {target}.")
    artifact_path = resolve_reported_path(artifact_report.get("path"))
    summary, artifact, validated_path, artifact_hash, review = (
        base.validate_artifact_and_summary(
            target,
            summary_path,
            artifact_path,
            manifest_path=manifest_path,
            development_metadata_path=manifest_path.parent / "development_metadata_v7.csv",
            regression_metadata_path=manifest_path.parent / "regression_metadata_v7.csv",
            run_root=paths.root,
        )
    )
    validate_fused_representation(target, artifact)
    if validated_path != artifact_path:
        raise ValueError(f"Resolved artifact path mismatch for {target}.")
    if artifact_report != _artifact_payload(artifact, artifact_path, artifact_hash, review):
        raise ValueError(f"Evaluation artifact provenance mismatch for {target}.")
    if result.get("training_summary_artifact") != summary.get("artifact"):
        raise ValueError(f"Evaluation training-artifact reference mismatch for {target}.")
    training_evidence = base.internal_test_evidence(summary)
    if training_evidence.get("passed") is not True:
        raise ValueError(f"Training internal-test evidence failed for {target}.")
    if result.get("training_internal_test") != training_evidence:
        raise ValueError(f"Evaluation internal-test evidence mismatch for {target}.")

    runtime_validation = result.get("runtime_validation")
    if not isinstance(runtime_validation, dict):
        raise ValueError(f"Evaluation has no runtime validation for {target}.")
    readiness = runtime_validation.get("provider_readiness")
    if not isinstance(readiness, dict) or readiness.get("ok") is not True:
        raise ValueError(f"Provider readiness evidence failed for {target}.")
    for name, expected in (
        ("model_family", artifact.model_family),
        ("model_id", artifact.model_id),
        ("model_revision", artifact.model_revision),
        ("tokenizer_revision", artifact.tokenizer_revision),
        ("preprocessing_sha256", artifact.preprocessing_sha256),
        ("layer", artifact.layer),
        ("pooling", artifact.pooling),
        ("feature_dim", artifact.feature_dim),
    ):
        if readiness.get(name) != expected:
            raise ValueError(
                f"Provider-readiness provenance mismatch for {target}, field {name!r}."
            )
    if runtime_validation.get("service_ready") is not True:
        raise ValueError(f"Service readiness evidence failed for {target}.")
    runtime_location_key = (
        "llava_runtime_model" if target == "llava05b" else "qwen_cache_dir"
    )
    if not isinstance(runtime_validation.get(runtime_location_key), str):
        raise ValueError(f"Runtime model/cache location is missing for {target}.")
    if result.get("traffic_mode") != "shadow":
        raise ValueError(f"Evaluation was not recorded in shadow mode for {target}.")
    if result.get("action_field_for_detector_evaluation") != "recommended_action":
        raise ValueError(f"Evaluation action-field contract changed for {target}.")

    rows_value = result.get("results")
    if not isinstance(rows_value, list) or not all(isinstance(row, dict) for row in rows_value):
        raise ValueError(f"Evaluation results must be a list of objects for {target}.")
    rows: list[dict[str, object]] = [dict(row) for row in rows_value]
    expected_rows = [
        (panel_name, metadata_row)
        for panel_name in ("regression", "text_led", "external_benign")
        for metadata_row in panels[panel_name]
    ]
    if len(rows) != len(expected_rows):
        raise ValueError(f"Evaluation result count mismatch for {target}.")

    metadata_fields = (
        "sample_id",
        "split",
        "group_id",
        "attack_style",
        "strategy",
        "source",
        "prompt_text",
        "image_text",
        "image_path",
        "image_sha256",
        "source_sample_id",
        "source_strategy",
        "paired_benign_sample_id",
        "hard_negative_category",
    )
    for sequence, (row, (panel_name, metadata_row)) in enumerate(
        zip(rows, expected_rows, strict=True), start=1
    ):
        sample_id = metadata_row["sample_id"]
        if row.get("sequence") != sequence or row.get("panel") != panel_name:
            raise ValueError(f"Evaluation row ordering mismatch at {target}/{sample_id}.")
        for name in metadata_fields:
            if row.get(name) != metadata_row[name]:
                raise ValueError(
                    f"Metadata mismatch at {target}/{sample_id}, field {name!r}."
                )
        if row.get("label_id") != int(metadata_row["label_id"]):
            raise ValueError(f"Label mismatch at {target}/{sample_id}.")
        label = int(metadata_row["label_id"])
        expected_label = "malicious" if label == 1 else "benign"
        expected_label_action = "block" if label == 1 else "allow"
        if row.get("expected_label") != expected_label:
            raise ValueError(f"Expected-label mismatch at {target}/{sample_id}.")
        if row.get("expected_recommended_action") != expected_label_action:
            raise ValueError(f"Expected-action mismatch at {target}/{sample_id}.")
        if row.get("render_style") != int(metadata_row["render_style"]):
            raise ValueError(f"Render-style mismatch at {target}/{sample_id}.")
        if row.get("hard_negative") != int(metadata_row.get("hard_negative", "0") or 0):
            raise ValueError(f"Hard-negative flag mismatch at {target}/{sample_id}.")
        if row.get("target") != target:
            raise ValueError(f"Target mismatch at {target}/{sample_id}.")
        for name, expected in (
            ("artifact_path", base.display_path(artifact_path)),
            ("artifact_sha256", artifact_hash),
            ("detector_source", artifact.source),
            ("model_family", artifact.model_family),
            ("model_id", artifact.model_id),
            ("model_revision", artifact.model_revision),
            ("tokenizer_revision", artifact.tokenizer_revision),
            ("preprocessing_sha256", artifact.preprocessing_sha256),
            ("layer", artifact.layer),
            ("pooling", artifact.pooling),
            ("feature_dim", artifact.feature_dim),
        ):
            if row.get(name) != expected:
                raise ValueError(
                    f"Artifact provenance mismatch at {target}/{sample_id}, field {name!r}."
                )
        _require_close(row.get("block_threshold"), artifact.threshold, "block threshold")
        _require_close(row.get("review_threshold"), review, "review threshold")
        score = _number(row.get("risk_score"), f"risk score for {target}/{sample_id}")
        verdict, action, uncertain = base.expected_decision_from_score(
            score, float(artifact.threshold), review
        )
        if row.get("static_expected_verdict") != verdict:
            raise ValueError(f"Static verdict mismatch at {target}/{sample_id}.")
        if row.get("static_expected_recommended_action") != action:
            raise ValueError(f"Static action mismatch at {target}/{sample_id}.")
        if row.get("verdict") != verdict or row.get("recommended_action") != action:
            raise ValueError(f"Runtime/static decision mismatch at {target}/{sample_id}.")
        if row.get("uncertain") is not uncertain:
            raise ValueError(f"Uncertainty mismatch at {target}/{sample_id}.")
        expected_reasons = [
            "score_above_threshold" if verdict == "malicious" else "score_below_threshold"
        ]
        if uncertain:
            expected_reasons.append("within_review_band")
        if row.get("reasons") != expected_reasons:
            raise ValueError(f"Decision-reason mismatch at {target}/{sample_id}.")
        if row.get("static_decision_matches") is not True:
            raise ValueError(f"Static integrity flag failed at {target}/{sample_id}.")
        _require_close(row.get("decision_threshold"), artifact.threshold, "decision threshold")
        _require_close(row.get("decision_review_threshold"), review, "decision review threshold")
        for name, expected in (
            ("decision_model_family", artifact.model_family),
            ("decision_model_id", artifact.model_id),
            ("decision_pooling", artifact.pooling),
            ("decision_detector_source", artifact.source),
        ):
            if row.get(name) != expected:
                raise ValueError(
                    f"Runtime provenance mismatch at {target}/{sample_id}, field {name!r}."
                )
        if row.get("error_type") is not None or row.get("error_detail") is not None:
            raise ValueError(f"Runtime error recorded at {target}/{sample_id}.")
        if row.get("request_id") != sample_id or row.get("modality") != "image_text":
            raise ValueError(f"Request identity mismatch at {target}/{sample_id}.")
        if row.get("traffic_mode") != "shadow" or row.get("enforcement_action") != "allow":
            raise ValueError(f"Shadow-mode action mismatch at {target}/{sample_id}.")
        if row.get("image_fingerprint_matches") is not True:
            raise ValueError(f"Image fingerprint mismatch at {target}/{sample_id}.")
        if row.get("decision_image_sha256") != [metadata_row["image_sha256"]]:
            raise ValueError(f"Recorded image fingerprint mismatch at {target}/{sample_id}.")
        if row.get("prompt_fingerprint_matches") is not True:
            raise ValueError(f"Prompt fingerprint mismatch at {target}/{sample_id}.")
        if row.get("accepted") is not (action == expected_label_action):
            raise ValueError(f"Row acceptance flag mismatch at {target}/{sample_id}.")

    expected_panels = {
        panel: base.panel_summary([row for row in rows if row["panel"] == panel])
        for panel in ("regression", "text_led", "external_benign")
    }
    if result.get("panels") != expected_panels:
        raise ValueError(f"Panel summary mismatch for {target}.")
    expected_acceptance = base.acceptance_report(rows, training_evidence)
    if result.get("acceptance") != expected_acceptance:
        raise ValueError(f"Acceptance summary mismatch for {target}.")
    if expected_acceptance.get("passed") is not True:
        raise ValueError(f"v7 promotion criteria failed for {target}.")

    validate_csv_copy(csv_path, rows)
    return {
        "target": target,
        "json": base.display_path(json_path),
        "json_sha256": base.sha256_file(json_path),
        "csv": base.display_path(csv_path),
        "csv_sha256": base.sha256_file(csv_path),
        "artifact_sha256": artifact_hash,
        "pooling": artifact.pooling,
        "feature_dim": artifact.feature_dim,
        "block_threshold": artifact.threshold,
        "review_threshold": review,
        "panels": expected_panels,
        "acceptance": expected_acceptance,
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
        from . import bordair_dual_evaluation_validation as dual

        targets = list(args.targets) if args.targets else list(TARGETS)
        output: list[dict[str, object]] = []
        for target in targets:
            report = dual.validate_evaluation(
                target=target,
                run_paths=paths,
                evaluation_json_path=(
                    evaluation_dir / f"{target}_dual_or_results.json"
                ),
                evaluation_csv_path=(
                    evaluation_dir / f"{target}_dual_or_results.csv"
                ),
                training_summary_path=(
                    training_dir / target / "dual_or" / "training_summary.json"
                ),
                manifest_path=manifest_path,
                final_metadata_path=final_metadata_path,
                text_led_metadata_path=text_led_metadata_path,
                external_benign_metadata_path=external_benign_metadata_path,
                cache_root=(
                    paths.feature_cache_directory / target / "evaluation"
                    if getattr(args, "cache_root", None) is None
                    else args.cache_root.expanduser().resolve()
                ),
            )
            base.atomic_write_json(
                evaluation_dir / f"{target}_dual_or_validation.json", report
            )
            output.append(report)
        print(json.dumps({"ok": True, "targets": output}, indent=2, sort_keys=True))
        return
    base.validate_tracked_v7_corpus(paths)
    targets = list(args.targets) if args.targets else list(TARGETS)
    output = [
        validate_target_evaluation(
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
        for target in targets
    ]
    print(json.dumps({"ok": True, "targets": output}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
