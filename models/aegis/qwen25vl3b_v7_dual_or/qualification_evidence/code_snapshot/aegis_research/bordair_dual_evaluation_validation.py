from __future__ import annotations

"""Independent validation of dual-head evaluation evidence."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import bordair_evaluation as base
from .bordair_dual import (
    combine_actions,
    head_action,
    qualification_cache_locator,
    score_pair,
    sha256_file,
)
from .bordair_dual_evaluation import (
    DUAL_CSV_FIELDS,
    acceptance_report,
    cache_contract,
    internal_test_evidence,
    panel_summary,
    validate_declared_snapshot_receipt,
    validate_exact_frozen_cache_report,
    validate_dual_training_summary,
)
from .bordair_paths import RunPaths


def _close(left: object, right: object) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _cache_scalar(data: Any, name: str) -> str:
    values = np.asarray(data[name]).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"cache field {name!r} must contain one value")
    return str(values[0])


def load_independently_validated_cache(
    *,
    cache_path: Path,
    panel: str,
    row: dict[str, str],
    panel_metadata_path: Path,
    panel_metadata_sha256: str,
    corpus_manifest_sha256: str,
    contract: Any,
    metadata_root: Path,
) -> np.ndarray:
    if not cache_path.is_file():
        raise FileNotFoundError(f"evaluation cache entry is missing: {cache_path}")
    image_path = (metadata_root / row["image_path"]).resolve()
    expected_strings = {
        "sample_id": row["sample_id"],
        "panel": panel,
        "caller_text_sha256": base.sha256_text(row["prompt_text"]),
        "image_sha256": row["image_sha256"],
        "image_path": row["image_path"],
        "panel_metadata_sha256": panel_metadata_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "model_family": contract.model_family,
        "model_id": contract.model_id,
        "model_revision": contract.model_revision,
        "tokenizer_revision": contract.tokenizer_revision,
        "preprocessing_sha256": contract.preprocessing_sha256,
        "pooling": contract.pooling,
    }
    if sha256_file(image_path) != row["image_sha256"]:
        raise ValueError(f"image bytes changed for {row['sample_id']}")
    if sha256_file(panel_metadata_path) != panel_metadata_sha256:
        raise ValueError(f"panel metadata bytes changed for {panel}")
    with np.load(cache_path, allow_pickle=False) as data:
        if set(data.files) != base.FEATURE_CACHE_FIELDS:
            raise ValueError(f"cache field schema changed for {row['sample_id']}")
        if int(_cache_scalar(data, "schema_version")) != base.FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported feature-cache schema")
        for name, expected in expected_strings.items():
            if _cache_scalar(data, name) != str(expected):
                raise ValueError(
                    f"cache provenance mismatch for {row['sample_id']} field {name}"
                )
        if int(_cache_scalar(data, "layer")) != int(contract.layer):
            raise ValueError("cache layer provenance mismatch")
        if int(_cache_scalar(data, "feature_dim")) != int(contract.feature_dim):
            raise ValueError("cache feature dimension provenance mismatch")
        embedding = np.asarray(data["embedding"], dtype=np.float64)
    if embedding.shape != (int(contract.feature_dim),):
        raise ValueError(f"cache embedding shape changed: {embedding.shape}")
    if not np.all(np.isfinite(embedding)):
        raise ValueError("cache embedding contains non-finite values")
    return embedding


def validate_recomputed_row(
    *,
    reported: dict[str, Any],
    metadata: dict[str, str],
    panel: str,
    pair: Any,
    embedding: np.ndarray,
    cache_path: Path,
    qualification_subject: str | None = None,
) -> dict[str, Any]:
    if reported.get("sample_id") != metadata["sample_id"]:
        raise ValueError("reported result sample order differs from metadata")
    if reported.get("panel") != panel:
        raise ValueError(f"reported panel differs for {metadata['sample_id']}")
    if int(reported.get("label_id")) != int(metadata["label_id"]):
        raise ValueError(f"reported label differs for {metadata['sample_id']}")
    scored = score_pair(pair, embedding)
    image_score = float(scored["image_scores"][0])
    text_score = float(scored["text_scores"][0])
    image_action = str(scored["image_actions"][0])
    text_action = str(scored["text_actions"][0])
    combined_action = str(scored["combined_actions"][0])
    image_entry = pair.manifest["artifacts"]["image"]
    text_entry = pair.manifest["artifacts"]["text"]
    expected_action = "block" if int(metadata["label_id"]) == 1 else "allow"
    accepted = combined_action == expected_action
    scalar_checks = {
        "image_head_score": image_score,
        "text_head_score": text_score,
        "image_head_block_threshold": float(pair.image_artifact.threshold),
        "image_head_review_threshold": image_entry["review_threshold"],
        "text_head_block_threshold": float(pair.text_artifact.threshold),
        "text_head_review_threshold": text_entry["review_threshold"],
    }
    for name, expected in scalar_checks.items():
        if not _close(reported.get(name), expected):
            raise ValueError(f"recomputed {name} differs for {metadata['sample_id']}")
    exact_checks = {
        "prompt_text": metadata["prompt_text"],
        "image_path": metadata["image_path"],
        "image_sha256": metadata["image_sha256"],
        "target": pair.manifest["target"],
        "expected_action": expected_action,
        "image_head_action": image_action,
        "text_head_action": text_action,
        "combined_action": combined_action,
        "accepted": accepted,
        "caller_text_sha256": base.sha256_text(metadata["prompt_text"]),
        "image_artifact_sha256": image_entry["sha256"],
        "text_artifact_sha256": text_entry["sha256"],
        "pair_identity_sha256": pair.manifest["pair_identity_sha256"],
        "runtime_detector_identity_sha256": pair.manifest[
            "runtime_detector_identity_sha256"
        ],
        "cache_path": (
            qualification_cache_locator(
                qualification_subject, panel, metadata["sample_id"]
            )
            if qualification_subject is not None
            else base.display_path(cache_path)
        ),
        "cache_sha256": sha256_file(cache_path),
    }
    for name, expected in exact_checks.items():
        if reported.get(name) != expected:
            raise ValueError(f"recomputed {name} differs for {metadata['sample_id']}")
    # Recompute actions a second way so score/action coupling is explicit.
    if image_action != head_action(
        image_score,
        block_threshold=float(pair.image_artifact.threshold),
        review_threshold=float(image_entry["review_threshold"]),
    ) or text_action != head_action(
        text_score,
        block_threshold=float(pair.text_artifact.threshold),
        review_threshold=float(text_entry["review_threshold"]),
    ):
        raise RuntimeError("independent per-head action recomputation disagrees")
    if combined_action != combine_actions(image_action, text_action):
        raise RuntimeError("independent OR action recomputation disagrees")
    return {
        **reported,
        "image_head_score": image_score,
        "text_head_score": text_score,
        "image_head_action": image_action,
        "text_head_action": text_action,
        "combined_action": combined_action,
        "accepted": accepted,
    }


def _csv_value(value: Any) -> str:
    return "" if value is None else str(value)


def validate_csv_copy(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != DUAL_CSV_FIELDS:
            raise ValueError("dual evaluation CSV columns changed")
        actual = list(reader)
    if len(actual) != len(rows):
        raise ValueError("dual evaluation CSV row count differs from JSON")
    for index, (csv_row, json_row) in enumerate(zip(actual, rows), start=1):
        for field in DUAL_CSV_FIELDS:
            if csv_row[field] != _csv_value(json_row.get(field)):
                raise ValueError(f"dual CSV/JSON mismatch at row {index}, field {field}")


def validate_evaluation(
    *,
    target: str,
    run_paths: RunPaths,
    evaluation_json_path: Path,
    evaluation_csv_path: Path,
    training_summary_path: Path,
    manifest_path: Path,
    final_metadata_path: Path,
    text_led_metadata_path: Path,
    external_benign_metadata_path: Path,
    cache_root: Path,
    qualification_workspace: Path | None = None,
) -> dict[str, Any]:
    qualification_snapshot = None
    if qualification_workspace is None:
        base.validate_tracked_v7_corpus(run_paths)
    else:
        qualification_snapshot = validate_declared_snapshot_receipt(
            workspace=qualification_workspace, target=target,
            run_paths=run_paths,
        )
    manifest, panels, metadata_info = base.validate_v7_manifest_and_panels(
        manifest_path,
        final_metadata_path,
        text_led_metadata_path,
        external_benign_metadata_path,
    )
    summary, pair = validate_dual_training_summary(
        target=target,
        summary_path=training_summary_path,
        run_paths=run_paths,
        manifest_path=manifest_path,
    )
    output = base.read_json(evaluation_json_path)
    if (
        output.get("schema_version") != 3
        or output.get("evaluation_version") != "v7"
        or output.get("artifact_mode") != "dual_or"
        or output.get("target") != target
    ):
        raise ValueError("dual evaluation identity is invalid")
    if output.get("dataset") != manifest["dataset"] or output.get(
        "dataset_commit"
    ) != manifest["dataset_commit"]:
        raise ValueError("dual evaluation dataset identity changed")
    if output.get("runtime_validation", {}).get(
        "qualification_snapshot"
    ) != qualification_snapshot:
        raise ValueError("dual evaluation qualification snapshot binding changed")
    reported_pair = output.get("detector_pair")
    if not isinstance(reported_pair, dict):
        raise ValueError("dual evaluation has no detector-pair binding")
    expected_pair_binding = {
        "manifest_sha256": sha256_file(pair.manifest_path),
        "pair_identity_sha256": pair.manifest["pair_identity_sha256"],
        "runtime_detector_identity_sha256": pair.manifest[
            "runtime_detector_identity_sha256"
        ],
        "composition": "or",
        "cache_ordering": ["text_tokens", "image_tokens"],
        "artifacts": pair.manifest["artifacts"],
    }
    for name, expected in expected_pair_binding.items():
        if reported_pair.get(name) != expected:
            raise ValueError(f"dual evaluation pair binding differs for {name}")
    reported_summary = output.get("training_summary")
    if not isinstance(reported_summary, dict) or reported_summary.get(
        "sha256"
    ) != sha256_file(training_summary_path):
        raise ValueError("dual evaluation training-summary binding changed")
    if reported_summary.get("training_identity_sha256") != summary[
        "training_identity"
    ]["sha256"]:
        raise ValueError("dual evaluation training identity changed")
    reported_rows = output.get("results")
    if not isinstance(reported_rows, list) or len(reported_rows) != 50:
        raise ValueError("dual evaluation must contain exactly 50 rows")
    contract = cache_contract(pair)
    corpus_hash = sha256_file(manifest_path)
    metadata_paths = {
        "regression": final_metadata_path,
        "text_led": text_led_metadata_path,
        "external_benign": external_benign_metadata_path,
    }
    recomputed: list[dict[str, Any]] = []
    position = 0
    cache_hashes: dict[str, str] = {}
    for panel in ("regression", "text_led", "external_benign"):
        metadata_hash = str(metadata_info[panel]["sha256"])
        for metadata_row in panels[panel]:
            reported = reported_rows[position]
            position += 1
            cache_path = cache_root / panel / f"{metadata_row['sample_id']}.npz"
            embedding = load_independently_validated_cache(
                cache_path=cache_path,
                panel=panel,
                row=metadata_row,
                panel_metadata_path=metadata_paths[panel],
                panel_metadata_sha256=metadata_hash,
                corpus_manifest_sha256=corpus_hash,
                contract=contract,
                metadata_root=manifest_path.parent,
            )
            recomputed.append(
                validate_recomputed_row(
                    reported=reported,
                    metadata=metadata_row,
                    panel=panel,
                    pair=pair,
                    embedding=embedding,
                    cache_path=cache_path,
                    qualification_subject=(
                        qualification_snapshot["qualification_subject_sha256"]
                        if qualification_snapshot is not None
                        else None
                    ),
                )
            )
            cache_hashes[f"{panel}/{metadata_row['sample_id']}.npz"] = sha256_file(
                cache_path
            )
    validate_csv_copy(evaluation_csv_path, reported_rows)
    training_evidence = internal_test_evidence(summary)
    recomputed_acceptance = acceptance_report(recomputed, training_evidence)
    if output.get("acceptance") != recomputed_acceptance:
        raise ValueError("dual evaluation acceptance evidence does not recompute")
    expected_panels = {
        panel: panel_summary([row for row in recomputed if row["panel"] == panel])
        for panel in ("regression", "text_led", "external_benign")
    }
    if output.get("panels") != expected_panels:
        raise ValueError("dual evaluation panel summaries do not recompute")
    cache_report = output.get("runtime_validation", {}).get("feature_cache")
    if not isinstance(cache_report, dict):
        raise ValueError("dual evaluation must record an enabled feature cache")
    exact_cache_statistics = validate_exact_frozen_cache_report(cache_report)
    if recomputed_acceptance["passed"] is not True:
        raise ValueError("dual evaluation did not pass frozen-panel gates")
    cache_tree_digest = base.sha256_text(
        json.dumps(cache_hashes, sort_keys=True, separators=(",", ":"))
    )
    return {
        "schema_version": 1,
        "validation_kind": "independent_dual_or_evaluation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "passed": True,
        "pair_identity_sha256": pair.manifest["pair_identity_sha256"],
        "runtime_detector_identity_sha256": pair.manifest[
            "runtime_detector_identity_sha256"
        ],
        "evidence": {
            "evaluation_json_sha256": sha256_file(evaluation_json_path),
            "evaluation_csv_sha256": sha256_file(evaluation_csv_path),
            "training_summary_sha256": sha256_file(training_summary_path),
            "pair_manifest_sha256": sha256_file(pair.manifest_path),
            "image_artifact_sha256": pair.manifest["artifacts"]["image"]["sha256"],
            "text_artifact_sha256": pair.manifest["artifacts"]["text"]["sha256"],
            "corpus_manifest_sha256": corpus_hash,
            "cache_entries": len(cache_hashes),
            "cache_tree_sha256": cache_tree_digest,
            "cache_statistics": exact_cache_statistics,
        },
        "acceptance": recomputed_acceptance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently validate a Bordair v7 dual-head evaluation."
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
    parser.add_argument("--qualification-workspace", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = RunPaths(args.run_root)
    evaluation_dir = paths.choose(args.evaluation_dir, paths.evaluation_directory)
    training_dir = paths.choose(args.training_dir, paths.training_directory)
    manifest_path = paths.corpus_input(args.manifest, paths.manifest, "--manifest")
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
    report = validate_evaluation(
        target=args.target,
        run_paths=paths,
        evaluation_json_path=evaluation_dir / f"{args.target}_dual_or_results.json",
        evaluation_csv_path=evaluation_dir / f"{args.target}_dual_or_results.csv",
        training_summary_path=(
            training_dir / args.target / "dual_or" / "training_summary.json"
        ),
        manifest_path=manifest_path,
        final_metadata_path=final_metadata_path,
        text_led_metadata_path=text_led_metadata_path,
        external_benign_metadata_path=external_benign_metadata_path,
        cache_root=(
            paths.feature_cache_directory / args.target / "evaluation"
            if args.cache_root is None
            else args.cache_root.expanduser().resolve()
        ),
        qualification_workspace=args.qualification_workspace,
    )
    output_path = (
        evaluation_dir / f"{args.target}_dual_or_validation.json"
        if args.output is None
        else args.output.expanduser().resolve()
    )
    base.atomic_write_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
