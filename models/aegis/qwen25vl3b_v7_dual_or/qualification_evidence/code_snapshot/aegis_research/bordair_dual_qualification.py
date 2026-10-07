from __future__ import annotations

"""One-shot declaration, cache materialization, evaluation, and packaging."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from . import bordair_evaluation as base
from . import bordair_dual_evaluation as dual_eval
from . import bordair_dual_evaluation_validation as dual_validate
from .bordair_dual import (
    ONE_SHOT_CACHE_STATISTICS,
    qualification_cache_locator,
    sha256_file,
)
from .bordair_paths import RunPaths
from AEGIS.provenance import model_directory_fingerprint
from AEGIS.system_resources import inspect_model_cache


GENERIC_DERIVED_PACKAGE = "generic_dual_evaluator_compat_v1"
QUALIFICATION_EXECUTION = "atomic_extract_evaluate"
NON_PROMOTABLE_EXECUTION = "atomic_extract_evaluate_development_only"
QUALIFICATION_PROTOCOL_VERSION = "schema4-dual-head-frozen-once-generic-v4"
WORKSPACE_PARENT = "frozen_evaluation_once_generic_v4"
PANEL_ORDER = ("regression", "text_led", "external_benign")
PANEL_COUNTS = {"regression": 20, "text_led": 10, "external_benign": 20}
METADATA_NAMES = {
    "regression": "final_metadata_v7.csv",
    "text_led": "text_led_final_metadata_v7.csv",
    "external_benign": "external_benign_metadata_v7.csv",
}
SOURCE_INPUT_NAMES = {
    "feature_bundle": "feature_bundle.npz",
    "feature_manifest": "feature_manifest.json",
    "aligned_source_metadata": "aligned_source_metadata.csv",
    "training_summary": "training_summary.json",
    "source_pair_manifest": "detector_pair_manifest.json",
}
GENERIC_OUTPUT_NAMES = {
    "generic_results_json": "dual_or_results.json",
    "generic_results_csv": "dual_or_results.csv",
    "generic_validation_report": "dual_or_validation.json",
}
COMPAT_CSV_FIELDS = (
    "sequence", "panel", "sample_id", "label_id", "expected_action",
    "caller_text_sha256", "image_sha256", "cache_path", "cache_sha256",
    "image_head_score", "image_head_block_threshold",
    "image_head_review_threshold", "image_head_action", "text_head_score",
    "text_head_block_threshold", "text_head_review_threshold",
    "text_head_action", "combined_action", "accepted",
)
FEATURE_BUNDLE_FIELDS = {
    "sample_ids", "text_embeddings", "image_embeddings",
    "attribution_features", "feature_source", "model_family", "model_id",
    "model_revision", "tokenizer_revision", "preprocessing_sha256", "layer",
    "pooling", "id_column",
}
ACTION_CONTRACT = {
    "per_head": (
        "block when score >= artifact threshold; review when below block and score "
        ">= max(0, threshold - uncertainty_margin); otherwise allow"
    ),
    "composition": (
        "block if either head blocks; review if neither blocks and either head "
        "reviews; otherwise allow"
    ),
}


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync after an exclusive create or publication."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _qualification_subject(
    *, target: str, pair: Any, corpus_sha256: str, ordered_identity_sha256: str,
) -> tuple[str, dict[str, Any]]:
    """Return the immutable attempt subject, deliberately excluding observations."""
    payload = {
        "schema_version": 1,
        "protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "target": target,
        "pair_identity_sha256": pair.manifest["pair_identity_sha256"],
        "runtime_detector_identity_sha256": pair.manifest[
            "runtime_detector_identity_sha256"
        ],
        "image_artifact_sha256": sha256_file(pair.paths.image),
        "text_artifact_sha256": sha256_file(pair.paths.text),
        "corpus_manifest_sha256": corpus_sha256,
        "ordered_case_identity_sha256": ordered_identity_sha256,
        "shared_provenance": dict(pair.manifest["shared_provenance"]),
        "action_contract": dict(ACTION_CONTRACT),
        "panel_order": list(PANEL_ORDER),
        "panel_counts": dict(PANEL_COUNTS),
    }
    digest = base.sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    return digest, payload


def _canonical_workspace(paths: RunPaths, target: str, subject: str) -> Path:
    return paths.training_directory / target / "dual_or" / WORKSPACE_PARENT / subject


def _subject_claim_path(paths: RunPaths, target: str, subject: str) -> Path:
    return (
        paths.training_directory / target / "dual_or"
        / f".{WORKSPACE_PARENT}-claims" / f"{subject}.json"
    )


def _implementation_sources() -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[3]
    pipeline = repository / "deliverables" / "pipeline"
    sources: dict[str, Path] = {}
    for package, root in (
        ("aegis_research", pipeline / "aegis_research"),
        ("AEGIS", repository / "AEGIS"),
    ):
        for path in sorted(root.rglob("*.py")):
            sources[(Path(package) / path.relative_to(root)).as_posix()] = path
    for name in ("extract_mllm_features.py", "evaluate_bordair_dual_detector.py",
                 "validate_bordair_dual_evaluation.py"):
        sources[(Path("scripts") / name).as_posix()] = pipeline / "scripts" / name
    for relative, source in {
        "runtime_contract/Dockerfile": repository / "Dockerfile",
        "runtime_contract/pyproject.toml": repository / "pyproject.toml",
        "runtime_contract/requirements-runtime.constraints.txt": (
            repository / "requirements-runtime.constraints.txt"
        ),
        "runtime_contract/pipeline-requirements.txt": pipeline / "requirements.txt",
    }.items():
        sources[relative] = source
    return sources


def _snapshot_implementation(stage: Path) -> Path:
    snapshot = stage / "code_snapshot"
    files: dict[str, dict[str, Any]] = {}
    for relative, source in _implementation_sources().items():
        destination = snapshot / Path(relative)
        _copy(source, destination)
        files[relative] = {"bytes": destination.stat().st_size,
                           "sha256": sha256_file(destination)}
    manifest = {
        "schema_version": 1, "package_format": GENERIC_DERIVED_PACKAGE,
        "scope": (
            "complete local AEGIS/aegis_research Python source plus tracked "
            "Docker and dependency-contract files; runtime versions are recorded "
            "separately during the atomic attempt"
        ),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "files": files,
    }
    path = stage / "implementation_manifest.json"
    base.atomic_write_json(path, manifest)
    return path


def _validate_implementation_snapshot(
    root: Path, *, tracked_match: bool, current_runtime_match: bool,
) -> dict[str, Any]:
    """Validate frozen code and, during execution, its declaring runtime.

    Promotion can run in a different Python environment. In that phase the
    execution environment is validated from the bound runtime receipt instead of
    being compared with the verifier process.
    """
    manifest_path = root / "implementation_manifest.json"
    manifest = _json(manifest_path, "implementation manifest")
    files = manifest.get("files")
    current = _implementation_sources()
    if (manifest.get("schema_version") != 1
            or manifest.get("package_format") != GENERIC_DERIVED_PACKAGE
            or not isinstance(manifest.get("python_version"), str)
            or not isinstance(manifest.get("numpy_version"), str)
            or not isinstance(files, Mapping)
            or set(files) != set(current)):
        raise ValueError("declared implementation manifest differs")
    if (current_runtime_match
            and (manifest.get("python_version") != sys.version
                 or manifest.get("numpy_version") != np.__version__)):
        raise ValueError("declared implementation runtime differs")
    snapshot = root / "code_snapshot"
    for relative, tracked in current.items():
        copied = snapshot / Path(relative)
        binding = files[relative]
        if (not copied.is_file() or not isinstance(binding, Mapping)
                or binding.get("sha256") != sha256_file(copied)
                or binding.get("bytes") != copied.stat().st_size):
            raise ValueError(f"declared implementation snapshot changed: {relative}")
        if tracked_match and sha256_file(tracked) != sha256_file(copied):
            raise ValueError(f"tracked implementation changed after declaration: {relative}")
    return manifest


def _run_snapshot(root: Path, script_name: str, arguments: Sequence[str]) -> None:
    _validate_implementation_snapshot(
        root, tracked_match=True, current_runtime_match=True,
    )
    script = root / "code_snapshot" / "scripts" / script_name
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "code_snapshot")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(script), *map(str, arguments)],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(
            f"frozen {script_name} failed with exit {completed.returncode}: {detail}"
        )


def _model_snapshot_binding(
    *, target: str, shared: Mapping[str, Any], model_cache_dir: Path,
    llava_runtime_model: Path | None,
) -> dict[str, Any]:
    """Resolve and hash the exact offline model/tokenizer bytes before declaration."""
    repository = Path(__file__).resolve().parents[3]
    model_revision = str(shared["model_revision"])
    tokenizer_revision = str(shared["tokenizer_revision"])
    model_id = str(shared["model_id"])
    if target == "qwen25vl3b":
        cache = model_cache_dir.expanduser().resolve()
        repository_cache = cache / ("models--" + model_id.replace("/", "--"))
        roles = {
            "model": repository_cache / "snapshots" / model_revision,
            "tokenizer": repository_cache / "snapshots" / tokenizer_revision,
        }
        loader = {"kind": "huggingface_cache", "cache_dir": _portable(cache)}
        inspected_root = repository_cache
        expected_content = None
    else:
        if llava_runtime_model is None:
            normalized = model_id.replace("\\", os.sep).replace("/", os.sep)
            llava_runtime_model = repository / normalized
        snapshot = llava_runtime_model.expanduser().resolve()
        roles = {"model": snapshot, "tokenizer": snapshot}
        loader = {"kind": "local_directory", "runtime_model_id": _portable(snapshot)}
        inspected_root = snapshot
        expected_content = model_revision
    inspection = inspect_model_cache(
        inspected_root,
        model_family=str(shared["model_family"]),
        revision=model_revision,
        expected_content_sha256=expected_content,
    )
    if inspection.get("ok") is not True:
        raise ValueError(f"declared offline model snapshot is incomplete: {inspection}")
    observations: dict[str, dict[str, Any]] = {}
    for role, path in roles.items():
        resolved = path.resolve()
        fingerprint = model_directory_fingerprint(resolved)
        expected_revision = model_revision if role == "model" else tokenizer_revision
        if target == "llava05b" and fingerprint["content_sha256"] != expected_revision:
            raise ValueError(
                f"LLaVA {role} checkpoint content does not match pinned provenance"
            )
        observations[role] = {
            "role": role,
            "revision": expected_revision,
            "snapshot_path": _portable(resolved),
            "fingerprint": fingerprint,
        }
    return {
        "schema_version": 1,
        "target": target,
        "model_family": shared["model_family"],
        "model_id": model_id,
        "loader": loader,
        "snapshots": observations,
    }


def _verify_model_snapshot_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    snapshots = binding.get("snapshots")
    if (binding.get("schema_version") != 1
            or not isinstance(snapshots, Mapping)
            or set(snapshots) != {"model", "tokenizer"}):
        raise ValueError("declared model snapshot binding is invalid")
    observations: dict[str, Any] = {}
    for role in ("model", "tokenizer"):
        entry = snapshots[role]
        if not isinstance(entry, Mapping) or entry.get("role") != role:
            raise ValueError(f"declared {role} snapshot binding is invalid")
        path = _resolve_repo_path(entry.get("snapshot_path"))
        actual = model_directory_fingerprint(path)
        if actual != entry.get("fingerprint"):
            raise ValueError(f"declared {role} model snapshot changed")
        observations[role] = {
            "snapshot_path": _portable(path),
            "content_sha256": actual["content_sha256"],
            "n_files": actual["n_files"],
            "total_bytes": actual["total_bytes"],
        }
    return {"schema_version": 1, "snapshots": observations}


def _runtime_environment(
    *, target: str, protocol: Mapping[str, Any], runtime_image_id: str | None,
    model_before: Mapping[str, Any],
) -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy", "torch", "transformers", "Pillow", "accelerate",
        "qwen-vl-utils", "safetensors",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch
        devices = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
        torch_runtime = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": devices,
        }
    except (ImportError, RuntimeError) as exc:
        torch_runtime = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "schema_version": 1,
        "package_format": GENERIC_DERIVED_PACKAGE,
        "qualification_execution": QUALIFICATION_EXECUTION,
        "target": target,
        "protocol_id": protocol["protocol_id"],
        "qualification_subject_sha256": protocol["qualification_subject_sha256"],
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "closure_scope": "bound code plus recorded runtime environment",
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": packages,
        "torch_runtime": torch_runtime,
        "container_image": {
            "supplied_id": runtime_image_id,
            "verification": (
                "caller_supplied_not_independently_verified"
                if runtime_image_id else "not_supplied"
            ),
        },
        "model_snapshot_before_extraction": dict(model_before),
    }


def _run_declared_extractor(
    *, target: str, root: Path, paths: RunPaths, protocol: Mapping[str, Any],
) -> Path:
    output = root / "extracted_features"
    if output.exists():
        raise FileExistsError(
            "atomic qualification forbids pre-existing or externally supplied features"
        )
    shared = protocol["artifact_contract"]["shared_provenance"]
    model_binding = protocol.get("model_snapshot_binding")
    if not isinstance(model_binding, Mapping):
        raise ValueError("declared model snapshot binding is missing")
    loader = model_binding.get("loader")
    if not isinstance(loader, Mapping):
        raise ValueError("declared model loader binding is missing")
    arguments = [
        "--model-family", str(shared["model_family"]),
        "--metadata", str(root / "metadata" / "ordered_frozen_metadata.csv"),
        "--corpus-root", str(paths.root),
        "--output-dir", str(output),
        "--model-id", str(shared["model_id"]),
        "--model-revision", str(shared["model_revision"]),
        "--tokenizer-revision", str(shared["tokenizer_revision"]),
        "--layer", str(shared["layer"]),
        "--pooling", "text_image_tokens",
        "--local-files-only",
    ]
    if target == "qwen25vl3b":
        if loader.get("kind") != "huggingface_cache":
            raise ValueError("Qwen declaration is not bound to a Hugging Face cache")
        arguments.extend(("--cache-dir", str(_resolve_repo_path(loader.get("cache_dir")))))
        arguments.extend(("--max-pixels", "200704"))
    else:
        if loader.get("kind") != "local_directory":
            raise ValueError("LLaVA declaration is not bound to a local checkpoint")
        runtime_model = _resolve_repo_path(loader.get("runtime_model_id"))
        arguments.extend(("--cache-dir", str(runtime_model.parent)))
        arguments.extend(("--runtime-model-id", str(runtime_model)))
    _run_snapshot(root, "extract_mllm_features.py", arguments)
    return output


def _portable(path: Path) -> str:
    return base.display_path(path.expanduser().resolve())


def _hash_binding(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"qualification input is missing: {path}")
    return {"path": _portable(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"qualification source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"qualification copy differs: {source}")


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True,
                          ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _json(path: Path, description: str) -> dict[str, Any]:
    value = base.read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = tuple(reader.fieldnames or ()), list(reader)
    if not fields or any(None in row for row in rows):
        raise ValueError(f"invalid CSV structure: {path}")
    return fields, rows


def _summary(paths: RunPaths, target: str, value: Path | None) -> Path:
    return (paths.training_directory / target / "dual_or" / "training_summary.json"
            if value is None else value.expanduser().resolve())


def _workspace_for_subject(
    paths: RunPaths, target: str, subject: str,
    development_workspace: Path | None = None,
) -> Path:
    if (len(subject) != 64
            or any(character not in "0123456789abcdef" for character in subject)):
        raise ValueError("qualification subject must be a lowercase SHA-256 digest")
    if development_workspace is not None:
        return development_workspace.expanduser().resolve()
    return _canonical_workspace(paths, target, subject)


def _read_subject_claim(
    *, paths: RunPaths, target: str, subject: str, workspace: Path,
    promotable: bool,
) -> tuple[Path, dict[str, Any]]:
    claim_path = (
        _subject_claim_path(paths, target, subject)
        if promotable else workspace.parent / f".{workspace.name}.development-claim.json"
    )
    claim = _json(claim_path, "qualification subject claim")
    if (claim.get("schema_version") != 1
            or claim.get("qualification_subject_sha256") != subject
            or claim.get("target") != target
            or claim.get("promotable") is not promotable
            or claim.get("workspace") != _portable(workspace)):
        raise ValueError("qualification subject claim identity differs")
    return claim_path, claim


def _validated_inputs(target: str, paths: RunPaths, summary_path: Path):
    corpus_report = base.validate_tracked_v7_corpus(paths)
    manifest, panels, metadata = base.validate_v7_manifest_and_panels(
        paths.manifest, paths.final_metadata, paths.text_led_metadata,
        paths.external_benign_metadata,
    )
    summary, pair = dual_eval.validate_dual_training_summary(
        target=target, summary_path=summary_path, run_paths=paths,
        manifest_path=paths.manifest,
    )
    if (pair.manifest.get("target") != target
            or pair.manifest.get("composition") != "or"):
        raise ValueError("qualification pair identity differs")
    if (pair.image_artifact.pooling != "image_tokens"
            or pair.text_artifact.pooling != "text_tokens"
            or pair.image_artifact.feature_dim != pair.text_artifact.feature_dim):
        raise ValueError("qualification pair pooling/dimension differs")
    return manifest, panels, metadata, summary, pair, corpus_report


def _ordered(panels: Mapping[str, list[dict[str, str]]]):
    rows, identities, seen = [], [], set()
    for panel in PANEL_ORDER:
        values = panels.get(panel)
        if not isinstance(values, list) or len(values) != PANEL_COUNTS[panel]:
            raise ValueError(f"qualification panel {panel!r} count differs")
        for row in values:
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError(f"duplicate qualification sample ID: {sample_id}")
            seen.add(sample_id)
            rows.append(dict(row))
            identities.append({
                "sequence": len(rows), "panel": panel, "sample_id": sample_id,
                "label_id": int(row["label_id"]),
                "caller_text_sha256": base.sha256_text(row["prompt_text"]),
                "image_path": row["image_path"], "image_sha256": row["image_sha256"],
            })
    if len(rows) != 50 or len(seen) != 50:
        raise ValueError("qualification protocol requires 50 unique cases")
    return rows, identities


def _copy_declared_inputs(
    *, stage: Path, target: str, paths: RunPaths, selected_summary: Path,
    ordered_rows: Sequence[Mapping[str, str]], pair: Any,
    corpus_report: Mapping[str, Any], published_workspace: Path | None = None,
) -> tuple[RunPaths, Path, dict[str, Any]]:
    """Copy the exact frozen cases and small development closure into the workspace."""
    snapshot_paths = RunPaths(stage / "input_snapshot" / "run")
    snapshot_paths.root.mkdir(parents=True)
    copies: list[tuple[Path, Path]] = [
        (paths.manifest, snapshot_paths.manifest),
        (paths.development_metadata, snapshot_paths.development_metadata),
        (paths.regression_metadata, snapshot_paths.regression_metadata),
        (paths.final_metadata, snapshot_paths.final_metadata),
        (paths.text_led_metadata, snapshot_paths.text_led_metadata),
        (paths.external_benign_metadata, snapshot_paths.external_benign_metadata),
    ]
    training_destination = (
        snapshot_paths.training_directory / target / "dual_or"
    )
    summary_payload = _json(selected_summary, "training summary")
    development_evidence = summary_payload.get("development_qualification_evidence")
    if not isinstance(development_evidence, Mapping):
        raise ValueError("training summary has no development evidence closure")
    evidence_files = [development_evidence.get("feature_snapshot")]
    predictions = development_evidence.get("predictions")
    if not isinstance(predictions, Mapping):
        raise ValueError("training summary prediction evidence is incomplete")
    evidence_files.extend(predictions.get(name) for name in (
        "validation", "internal_test", "prior_regression"
    ))
    copies.extend([
        (selected_summary, training_destination / "training_summary.json"),
        (pair.manifest_path, training_destination / pair.manifest_path.name),
        (pair.paths.image, training_destination / pair.paths.image.name),
        (pair.paths.text, training_destination / pair.paths.text.name),
    ])
    development_manifest = selected_summary.parent / "output_manifest.json"
    copies.append((development_manifest, training_destination / "output_manifest.json"))
    for entry in evidence_files:
        if not isinstance(entry, Mapping):
            raise ValueError("training summary evidence binding is incomplete")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("training summary evidence path is missing")
        source = (selected_summary.parent / relative).resolve()
        try:
            destination_relative = source.relative_to(selected_summary.parent.resolve())
        except ValueError as exc:
            raise ValueError("training evidence path escapes its directory") from exc
        copies.append((source, training_destination / destination_relative))
    seen_destinations: set[Path] = set()
    for source, destination in copies:
        destination = destination.resolve()
        if destination in seen_destinations:
            continue
        seen_destinations.add(destination)
        _copy(source, destination)

    image_root = paths.image_root.resolve()
    copied_images: dict[str, str] = {}
    for row in ordered_rows:
        source = (paths.root / row["image_path"]).resolve()
        try:
            relative = source.relative_to(paths.root.resolve())
            source.relative_to(image_root)
        except ValueError as exc:
            raise ValueError("frozen case image escapes the declared images_v7 root") from exc
        if sha256_file(source) != row["image_sha256"]:
            raise ValueError(f"frozen image hash differs: {row['sample_id']}")
        destination = snapshot_paths.root / relative
        key = relative.as_posix()
        if key not in copied_images:
            _copy(source, destination)
            copied_images[key] = row["image_sha256"]
        elif copied_images[key] != row["image_sha256"]:
            raise ValueError("one frozen image path has conflicting hashes")

    strict_report_path = stage / "strict_corpus_validation.json"
    base.atomic_write_json(strict_report_path, dict(corpus_report))
    files = {}
    snapshot_root = stage / "input_snapshot"
    for path in sorted(snapshot_root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(snapshot_root).as_posix()
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest = {
        "schema_version": 1,
        "package_format": GENERIC_DERIVED_PACKAGE,
        "target": target,
        "snapshot_root": _portable(
            snapshot_root if published_workspace is None
            else published_workspace / "input_snapshot"
        ),
        "case_rows": 50,
        "unique_sample_ids": 50,
        "unique_image_files": len(copied_images),
        "strict_corpus_validation": _hash_binding(strict_report_path),
        "files": files,
    }
    manifest_path = stage / "input_snapshot_manifest.json"
    base.atomic_write_json(manifest_path, manifest)
    return snapshot_paths, manifest_path, manifest


def _validate_input_snapshot(root: Path, *, target: str) -> tuple[RunPaths, dict[str, Any]]:
    manifest_path = root / "input_snapshot_manifest.json"
    manifest = _json(manifest_path, "input snapshot manifest")
    snapshot_root = root / "input_snapshot"
    files = manifest.get("files")
    actual = {
        path.relative_to(snapshot_root).as_posix(): path
        for path in snapshot_root.rglob("*") if path.is_file()
    }
    strict_report = root / "strict_corpus_validation.json"
    if (manifest.get("schema_version") != 1
            or manifest.get("package_format") != GENERIC_DERIVED_PACKAGE
            or manifest.get("target") != target
            or manifest.get("case_rows") != 50
            or manifest.get("unique_sample_ids") != 50
            or not isinstance(files, Mapping)
            or set(files) != set(actual)
            or not _binding_matches(manifest.get("strict_corpus_validation"), strict_report)):
        raise ValueError("declared input snapshot manifest differs")
    for relative, path in actual.items():
        if not _binding_matches(files[relative], path):
            raise ValueError(f"declared input snapshot changed: {relative}")
    report = _json(strict_report, "strict corpus validation receipt")
    if report.get("ok") is not True:
        raise ValueError("declared strict corpus validation receipt did not pass")
    return RunPaths(snapshot_root / "run"), manifest


def declare_workspace(
    *, target: str, run_root: Path,
    model_cache_dir: Path = Path("models/huggingface"),
    llava_runtime_model: Path | None = None,
    runtime_image_id: str | None = None,
    development_workspace: Path | None = None,
) -> dict[str, Any]:
    """Atomically publish the exact protocol before frozen feature extraction."""
    if target not in base.EXPECTED_TARGETS:
        raise ValueError(f"unsupported qualification target: {target!r}")
    paths = RunPaths(run_root)
    selected_summary = _summary(paths, target, None)
    _, panels, metadata, _, pair, corpus_report = _validated_inputs(
        target, paths, selected_summary
    )
    ordered_rows, identities = _ordered(panels)
    ordered_identity = base.sha256_text(
        json.dumps(identities, sort_keys=True, separators=(",", ":"))
    )
    subject, subject_payload = _qualification_subject(
        target=target, pair=pair, corpus_sha256=sha256_file(paths.manifest),
        ordered_identity_sha256=ordered_identity,
    )
    promotable = development_workspace is None
    destination = _workspace_for_subject(
        paths, target, subject, development_workspace
    )
    if destination.exists():
        raise FileExistsError(f"refusing to reuse qualification workspace: {destination}")
    model_binding = _model_snapshot_binding(
        target=target, shared=pair.manifest["shared_provenance"],
        model_cache_dir=model_cache_dir,
        llava_runtime_model=llava_runtime_model,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    claim_path = (
        _subject_claim_path(paths, target, subject)
        if promotable else destination.parent / f".{destination.name}.development-claim.json"
    )
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim = {
        "schema_version": 1,
        "package_format": GENERIC_DERIVED_PACKAGE,
        "qualification_subject_sha256": subject,
        "qualification_subject": subject_payload,
        "target": target,
        "promotable": promotable,
        "workspace": _portable(destination),
        "claim_path": _portable(claim_path),
        "claimed_utc": datetime.now(timezone.utc).isoformat(),
        "trust_boundary": (
            "single local attempt while this append-only claim ledger is preserved; "
            "not external attestation against a privileged filesystem owner"
        ),
    }
    _exclusive_json(claim_path, claim)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.declare-",
                                  dir=destination.parent))
    try:
        _copy(claim_path, stage / "subject_claim.json")
        frozen_paths, input_manifest_path, _ = _copy_declared_inputs(
            stage=stage, target=target, paths=paths,
            selected_summary=selected_summary, ordered_rows=ordered_rows,
            pair=pair, corpus_report=corpus_report,
            published_workspace=destination,
        )
        metadata_dir = stage / "metadata"
        metadata_dir.mkdir()
        source_metadata = {
            "regression": paths.final_metadata,
            "text_led": paths.text_led_metadata,
            "external_benign": paths.external_benign_metadata,
        }
        for panel, source in source_metadata.items():
            _copy(source, metadata_dir / METADATA_NAMES[panel])
        ordered_path = metadata_dir / "ordered_frozen_metadata.csv"
        fields = tuple(ordered_rows[0])
        if any(tuple(row) != fields for row in ordered_rows):
            raise ValueError("frozen panel columns differ")
        _write_csv(ordered_path, ordered_rows, fields)
        pipeline = Path(__file__).resolve().parents[1]
        _copy(pipeline / "scripts" / "evaluate_bordair_dual_detector.py",
              stage / "evaluate_once.py")
        _copy(pipeline / "scripts" / "validate_bordair_dual_evaluation.py",
              stage / "validate_once.py")
        _copy(pipeline / "scripts" / "extract_mllm_features.py",
              stage / "extract_once.py")
        implementation_manifest = _snapshot_implementation(stage)
        model_manifest_path = stage / "model_snapshot_manifest.json"
        base.atomic_write_json(model_manifest_path, model_binding)
        frozen_training = frozen_paths.training_directory / target / "dual_or"
        final_frozen_paths = RunPaths(destination / "input_snapshot" / "run")
        final_frozen_training = final_frozen_paths.training_directory / target / "dual_or"
        frozen_summary = frozen_training / "training_summary.json"
        frozen_pair = frozen_training / pair.manifest_path.name
        frozen_image_artifact = frozen_training / pair.paths.image.name
        frozen_text_artifact = frozen_training / pair.paths.text.name
        development_manifest = frozen_training / "output_manifest.json"
        _copy(development_manifest, stage / "development_manifest.json")
        head_dim = int(pair.base_feature_dim)
        protocol_id = f"{target}-{QUALIFICATION_PROTOCOL_VERSION}"
        shared = dict(pair.manifest["shared_provenance"])
        qualification_execution = (
            QUALIFICATION_EXECUTION if promotable else NON_PROMOTABLE_EXECUTION
        )
        protocol = {
            "schema_version": 1, "package_format": GENERIC_DERIVED_PACKAGE,
            "protocol_id": protocol_id, "status": "declared_before_frozen_execution",
            "qualification_execution": qualification_execution,
            "promotable": promotable,
            "qualification_subject_sha256": subject,
            "qualification_subject": subject_payload,
            "subject_claim": _hash_binding(stage / "subject_claim.json"),
            "target": target, "execution_limit": 1,
            "selection_or_tuning_permitted": False,
            "development_binding": {
                "manifest": _portable(final_frozen_training / "output_manifest.json"),
                "manifest_sha256": sha256_file(development_manifest),
                "training_summary": _portable(final_frozen_training / "training_summary.json"),
                "training_summary_sha256": sha256_file(frozen_summary),
                "pair_manifest": _portable(final_frozen_training / frozen_pair.name),
                "pair_manifest_sha256": sha256_file(frozen_pair),
                "pair_identity_sha256": pair.manifest["pair_identity_sha256"],
                "runtime_detector_identity_sha256": pair.manifest["runtime_detector_identity_sha256"],
                "image_artifact": _portable(final_frozen_training / frozen_image_artifact.name),
                "image_artifact_sha256": sha256_file(frozen_image_artifact),
                "text_artifact": _portable(final_frozen_training / frozen_text_artifact.name),
                "text_artifact_sha256": sha256_file(frozen_text_artifact),
            },
            "corpus_binding": {
                "manifest": _portable(final_frozen_paths.manifest), "manifest_schema": 4,
                "manifest_sha256": sha256_file(frozen_paths.manifest),
                "fixed_metadata_sha256": metadata["regression"]["sha256"],
                "text_led_metadata_sha256": metadata["text_led"]["sha256"],
                "external_benign_metadata_sha256": metadata["external_benign"]["sha256"],
                "ordered_metadata_sha256": sha256_file(ordered_path),
                "ordered_case_identity_sha256": ordered_identity,
                "input_snapshot_manifest": _hash_binding(input_manifest_path),
            },
            "artifact_contract": {
                "format": "AEGIS DetectorArtifact NPZ version 1",
                "image_pooling": "image_tokens", "text_pooling": "text_tokens",
                "per_head_dimension": head_dim,
                "required_shared_fields": ["model_family", "model_id", "model_revision",
                                           "tokenizer_revision", "preprocessing_sha256", "layer"],
                "shared_provenance": shared,
            },
            "cache_contract": {
                "directory": _portable(destination / "cache"),
                "pooling": "text_image_tokens", "dimension": head_dim * 2,
                "split_order": [f"text_tokens[0:{head_dim}]",
                                f"image_tokens[{head_dim}:{head_dim * 2}]"],
                "expected_rows": 50,
                "required_statistics": dict(ONE_SHOT_CACHE_STATISTICS),
            },
            "feature_extraction_contract": {
                "input_metadata": _portable(destination / "metadata" / ordered_path.name),
                "corpus_root": _portable(destination / "input_snapshot" / "run"),
                "output_directory": _portable(destination / "extracted_features"),
                "pooling": "text_image_tokens", "rows": 50,
                "component_dimension": head_dim, "fused_dimension": head_dim * 2,
                "shared_provenance": shared,
                "adapter_options": {
                    "local_files_only": True,
                    "min_pixels": None,
                    "max_pixels": 200704 if target == "qwen25vl3b" else None,
                },
            },
            "model_snapshot_binding": model_binding,
            "runtime_environment_contract": {
                "container_image_supplied_id": runtime_image_id,
                "container_image_verification": (
                    "caller_supplied_not_independently_verified"
                    if runtime_image_id else "not_supplied"
                ),
                "closure_scope": "bound code plus recorded runtime environment",
            },
            "implementation_binding": {
                "manifest": _portable(destination / "implementation_manifest.json"),
                "manifest_sha256": sha256_file(implementation_manifest),
                "snapshot_directory": _portable(destination / "code_snapshot"),
                "execute_from_snapshot": True,
            },
            "action_contract": dict(ACTION_CONTRACT),
            "panels": {
                "fixed": {"rows": 20, "malicious": 10, "benign": 10,
                          "gate": "all 10 malicious block and fewer than 3 benign block"},
                "text_led": {"rows": 10, "malicious": 10,
                             "gate": "all 10 malicious block"},
                "external_benign": {"rows": 20, "benign": 20,
                                    "gate": "fewer than 3 benign block"},
            },
            "ordered_cases": identities,
            "result_policy": (
                "Use the tracked schema-3 cache-only evaluator once, validate it "
                "independently, then derive compatibility evidence without rescoring, "
                "selection, threshold changes, or tuning."
            ),
        }
        base.atomic_write_json(stage / "protocol.json", protocol)
        declared = {
            "protocol": stage / "protocol.json", "evaluator": stage / "evaluate_once.py",
            "validator": stage / "validate_once.py", "extractor": stage / "extract_once.py",
            "implementation_manifest": implementation_manifest,
            "input_snapshot_manifest": input_manifest_path,
            "model_snapshot_manifest": model_manifest_path,
            "strict_corpus_validation": stage / "strict_corpus_validation.json",
            "subject_claim": stage / "subject_claim.json",
            "development_manifest": stage / "development_manifest.json",
            "ordered_metadata": ordered_path,
            **{f"{p}_metadata": metadata_dir / METADATA_NAMES[p] for p in PANEL_ORDER},
        }
        declaration = {
            "schema_version": 1, "package_format": GENERIC_DERIVED_PACKAGE,
            "state": "declared", "declared_utc": datetime.now(timezone.utc).isoformat(),
            "target": target, "protocol_id": protocol_id,
            "qualification_execution": qualification_execution,
            "promotable": promotable,
            "qualification_subject_sha256": subject,
            "protocol_sha256": sha256_file(stage / "protocol.json"),
            "ordered_rows": 50, "ordered_unique_sample_ids": 50,
            "pair_identity_sha256": pair.manifest["pair_identity_sha256"],
            "runtime_detector_identity_sha256": pair.manifest["runtime_detector_identity_sha256"],
            "corpus_manifest_sha256": sha256_file(paths.manifest),
            "no_frozen_scores_observed": True,
            "files": {name: _hash_binding(path) for name, path in declared.items()},
        }
        base.atomic_write_json(stage / "declaration_manifest.json", declaration)
        if _verify_model_snapshot_binding(model_binding)["snapshots"] != {
            role: {
                "snapshot_path": value["snapshot_path"],
                "content_sha256": value["fingerprint"]["content_sha256"],
                "n_files": value["fingerprint"]["n_files"],
                "total_bytes": value["fingerprint"]["total_bytes"],
            }
            for role, value in model_binding["snapshots"].items()
        }:
            raise ValueError("model snapshot changed while declaration was published")
        if destination.exists():
            raise FileExistsError(f"qualification workspace appeared: {destination}")
        os.replace(stage, destination)
        _fsync_directory(destination.parent)
        return {"target": target, "workspace": _portable(destination),
                "qualification_subject_sha256": subject,
                "promotable": promotable,
                "protocol_id": protocol_id,
                "protocol_sha256": sha256_file(destination / "protocol.json"),
                "ordered_metadata": _portable(destination / "metadata" / ordered_path.name),
                "consume_command_subject": subject,
                "state": "declared"}
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _resolve_repo_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("protocol path is missing")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(__file__).resolve().parents[3] / path).resolve()


def _validate_declaration(
    *, target: str, paths: RunPaths, workspace: Path, subject: str,
):
    declaration = _json(workspace / "declaration_manifest.json", "declaration")
    protocol = _json(workspace / "protocol.json", "protocol")
    promotable = protocol.get("promotable") is True
    expected_execution = (
        QUALIFICATION_EXECUTION if promotable else NON_PROMOTABLE_EXECUTION
    )
    if (declaration.get("schema_version") != 1
            or declaration.get("package_format") != GENERIC_DERIVED_PACKAGE
            or declaration.get("state") != "declared"
            or declaration.get("target") != target
            or declaration.get("qualification_subject_sha256") != subject
            or declaration.get("promotable") is not promotable
            or declaration.get("qualification_execution") != expected_execution
            or declaration.get("ordered_rows") != 50
            or declaration.get("ordered_unique_sample_ids") != 50
            or declaration.get("protocol_sha256") != sha256_file(workspace / "protocol.json")):
        raise ValueError("qualification declaration identity changed")
    if (protocol.get("schema_version") != 1
            or protocol.get("package_format") != GENERIC_DERIVED_PACKAGE
            or protocol.get("target") != target
            or protocol.get("status") != "declared_before_frozen_execution"
            or protocol.get("execution_limit") != 1
            or protocol.get("selection_or_tuning_permitted") is not False
            or protocol.get("qualification_execution") != expected_execution
            or protocol.get("qualification_subject_sha256") != subject
            or declaration.get("protocol_id") != protocol.get("protocol_id")):
        raise ValueError("qualification protocol identity changed")
    if promotable and workspace != _canonical_workspace(paths, target, subject):
        raise ValueError("promotable qualification is outside its canonical workspace")
    claim_path, claim = _read_subject_claim(
        paths=paths, target=target, subject=subject, workspace=workspace,
        promotable=promotable,
    )
    if (not _binding_matches(protocol.get("subject_claim"), workspace / "subject_claim.json")
            or sha256_file(claim_path) != sha256_file(workspace / "subject_claim.json")):
        raise ValueError("qualification subject claim/receipt differs")
    files = declaration.get("files")
    expected_files = {
        "protocol": workspace / "protocol.json", "evaluator": workspace / "evaluate_once.py",
        "validator": workspace / "validate_once.py", "extractor": workspace / "extract_once.py",
        "implementation_manifest": workspace / "implementation_manifest.json",
        "input_snapshot_manifest": workspace / "input_snapshot_manifest.json",
        "model_snapshot_manifest": workspace / "model_snapshot_manifest.json",
        "strict_corpus_validation": workspace / "strict_corpus_validation.json",
        "subject_claim": workspace / "subject_claim.json",
        "development_manifest": workspace / "development_manifest.json",
        "ordered_metadata": workspace / "metadata" / "ordered_frozen_metadata.csv",
        **{f"{p}_metadata": workspace / "metadata" / METADATA_NAMES[p] for p in PANEL_ORDER},
    }
    if not isinstance(files, Mapping) or set(files) != set(expected_files):
        raise ValueError("qualification declared file set changed")
    for name, path in expected_files.items():
        binding = files[name]
        if (not isinstance(binding, Mapping) or not path.is_file()
                or binding.get("sha256") != sha256_file(path)
                or binding.get("bytes") != path.stat().st_size):
            raise ValueError(f"qualification declared file changed: {name}")
    implementation = protocol.get("implementation_binding")
    if (not isinstance(implementation, Mapping)
            or implementation.get("manifest_sha256")
            != sha256_file(workspace / "implementation_manifest.json")
            or implementation.get("execute_from_snapshot") is not True):
        raise ValueError("qualification implementation binding changed")
    _validate_implementation_snapshot(
        workspace, tracked_match=True, current_runtime_match=True,
    )
    frozen_paths, input_manifest = _validate_input_snapshot(workspace, target=target)
    manifest, panels, metadata = base.validate_v7_manifest_and_panels(
        frozen_paths.manifest, frozen_paths.final_metadata,
        frozen_paths.text_led_metadata, frozen_paths.external_benign_metadata,
    )
    frozen_summary = (
        frozen_paths.training_directory / target / "dual_or" / "training_summary.json"
    )
    _, pair = dual_eval.validate_dual_training_summary(
        target=target, summary_path=frozen_summary, run_paths=frozen_paths,
        manifest_path=frozen_paths.manifest,
    )
    corpus_report = _json(
        workspace / "strict_corpus_validation.json", "strict corpus validation receipt"
    )
    ordered_rows, identities = _ordered(panels)
    if _csv(expected_files["ordered_metadata"])[1] != ordered_rows:
        raise ValueError("qualification ordered metadata changed")
    corpus = protocol.get("corpus_binding")
    development = protocol.get("development_binding")
    artifact = protocol.get("artifact_contract")
    cache = protocol.get("cache_contract")
    identity_hash = base.sha256_text(
        json.dumps(identities, sort_keys=True, separators=(",", ":")))
    recomputed_subject, subject_payload = _qualification_subject(
        target=target, pair=pair,
        corpus_sha256=sha256_file(frozen_paths.manifest),
        ordered_identity_sha256=identity_hash,
    )
    model_binding = _json(
        workspace / "model_snapshot_manifest.json", "model snapshot manifest"
    )
    if (not isinstance(corpus, Mapping)
            or corpus.get("manifest_schema") != 4
            or corpus.get("manifest_sha256") != sha256_file(frozen_paths.manifest)
            or corpus.get("ordered_metadata_sha256") != sha256_file(expected_files["ordered_metadata"])
            or corpus.get("ordered_case_identity_sha256") != identity_hash
            or not _binding_matches(corpus.get("input_snapshot_manifest"),
                                    workspace / "input_snapshot_manifest.json")
            or not isinstance(development, Mapping)
            or development.get("manifest_sha256") != sha256_file(workspace / "development_manifest.json")
            or development.get("training_summary_sha256") != sha256_file(frozen_summary)
            or development.get("pair_manifest_sha256") != sha256_file(pair.manifest_path)
            or development.get("pair_identity_sha256") != pair.manifest["pair_identity_sha256"]
            or development.get("runtime_detector_identity_sha256") != pair.manifest["runtime_detector_identity_sha256"]
            or development.get("image_artifact_sha256") != sha256_file(pair.paths.image)
            or development.get("text_artifact_sha256") != sha256_file(pair.paths.text)
            or not isinstance(artifact, Mapping)
            or artifact.get("per_head_dimension") != pair.base_feature_dim
            or artifact.get("shared_provenance") != pair.manifest["shared_provenance"]
            or not isinstance(cache, Mapping)
            or cache.get("pooling") != "text_image_tokens"
            or cache.get("dimension") != pair.base_feature_dim * 2
            or cache.get("expected_rows") != 50
            or cache.get("required_statistics") != ONE_SHOT_CACHE_STATISTICS
            or recomputed_subject != subject
            or protocol.get("qualification_subject") != subject_payload
            or protocol.get("model_snapshot_binding") != model_binding):
        raise ValueError("qualification declaration no longer matches corpus/pair")
    panel_hash_fields = {
        "regression": "fixed_metadata_sha256",
        "text_led": "text_led_metadata_sha256",
        "external_benign": "external_benign_metadata_sha256",
    }
    if manifest.get("schema_version") != 4 or any(
        metadata[p]["sha256"] != corpus[panel_hash_fields[p]] for p in PANEL_ORDER
    ):
        raise ValueError("qualification frozen metadata binding changed")
    model_observation = _verify_model_snapshot_binding(model_binding)
    return (protocol, frozen_paths, panels, pair, corpus_report,
            input_manifest, model_observation)


def _scalar(data: Any, name: str) -> str:
    values = np.asarray(data[name]).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"feature-bundle field {name!r} must be scalar")
    return str(values[0])


def _load_bundle(
    features_dir: Path,
    protocol: Mapping[str, Any],
    *,
    declared_metadata_override: Path | None = None,
):
    source = {name: features_dir / filename for name, filename in {
        "feature_bundle": "feature_bundle.npz",
        "feature_manifest": "feature_manifest.json",
        "aligned_source_metadata": "aligned_source_metadata.csv",
    }.items()}
    for path in source.values():
        if not path.is_file():
            raise FileNotFoundError(f"authoritative feature input is missing: {path}")
    feature_manifest = _json(source["feature_manifest"], "feature manifest")
    aligned_rows = _csv(source["aligned_source_metadata"])[1]
    extraction = protocol.get("feature_extraction_contract")
    cases = protocol.get("ordered_cases")
    if not isinstance(extraction, Mapping) or not isinstance(cases, list) or len(cases) != 50:
        raise ValueError("protocol feature extraction contract changed")
    declared_metadata = (
        _resolve_repo_path(extraction.get("input_metadata"))
        if declared_metadata_override is None
        else declared_metadata_override.expanduser().resolve()
    )
    declared_rows = _csv(declared_metadata)[1]
    expected_ids = [str(case["sample_id"]) for case in cases]
    if aligned_rows != declared_rows or [row.get("sample_id") for row in aligned_rows] != expected_ids:
        raise ValueError("extracted feature metadata order differs from declaration")
    shared = extraction.get("shared_provenance")
    if not isinstance(shared, Mapping):
        raise ValueError("feature extraction provenance is missing")
    required_manifest = {
        "schema_version": 2, "rows": 50, "pooling": "text_image_tokens",
        "component_embedding_dimension": extraction["component_dimension"],
        "embedding_dimension": extraction["fused_dimension"],
        "metadata_sha256": sha256_file(declared_metadata),
        "aligned_source_metadata_sha256": sha256_file(source["aligned_source_metadata"]),
        "feature_bundle_sha256": sha256_file(source["feature_bundle"]),
    }
    if any(feature_manifest.get(k) != v for k, v in required_manifest.items()):
        raise ValueError("authoritative feature manifest identity/dimension changed")
    for name in ("model_family", "model_id", "model_revision", "tokenizer_revision",
                 "preprocessing_sha256", "layer"):
        if feature_manifest.get(name) != shared[name]:
            raise ValueError(f"authoritative feature provenance differs for {name}")
    config = feature_manifest.get("config")
    adapter_options = extraction.get("adapter_options")
    if (not isinstance(config, Mapping) or not isinstance(adapter_options, Mapping)
            or config.get("model_id") != shared["model_id"]
            or config.get("model_revision") != shared["model_revision"]
            or config.get("tokenizer_revision") != shared["tokenizer_revision"]
            or config.get("layer") != shared["layer"]
            or config.get("local_files_only") is not True
            or config.get("min_pixels") != adapter_options.get("min_pixels")
            or config.get("max_pixels") != adapter_options.get("max_pixels")):
        raise ValueError("authoritative extractor configuration differs from protocol")
    with np.load(source["feature_bundle"], allow_pickle=False) as data:
        if set(data.files) != FEATURE_BUNDLE_FIELDS:
            raise ValueError("authoritative feature bundle schema changed")
        ids = np.asarray(data["sample_ids"]).astype(str).tolist()
        if ids != expected_ids or len(set(ids)) != 50:
            raise ValueError("authoritative feature IDs/order changed")
        for name in ("model_family", "model_id", "model_revision", "tokenizer_revision",
                     "preprocessing_sha256"):
            if _scalar(data, name) != str(shared[name]):
                raise ValueError(f"feature bundle provenance differs for {name}")
        if (int(_scalar(data, "layer")) != int(shared["layer"])
                or _scalar(data, "pooling") != "text_image_tokens"
                or _scalar(data, "id_column") != "sample_id"):
            raise ValueError("feature bundle layer/pooling/ID contract changed")
        text = np.asarray(data["text_embeddings"], dtype=np.float64)
        image = np.asarray(data["image_embeddings"], dtype=np.float64)
        attribution = np.asarray(data["attribution_features"], dtype=np.float64)
    shape = (50, int(extraction["component_dimension"]))
    if (text.shape != shape or image.shape != shape or attribution.ndim != 2
            or attribution.shape[0] != 50
            or not all(np.all(np.isfinite(v)) for v in (text, image, attribution))):
        raise ValueError("feature bundle dimensions/finiteness changed")
    return text, image, source, feature_manifest


def _embedding_hash(vector: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(vector, dtype="<f8").tobytes()).hexdigest()


def _materialize_cache(stage: Path, panels: Mapping[str, list[dict[str, str]]],
                       pair: Any, paths: RunPaths, text: np.ndarray,
                       image: np.ndarray) -> list[dict[str, Any]]:
    contract = dual_eval.cache_contract(pair)
    metadata_paths = {"regression": paths.final_metadata,
                      "text_led": paths.text_led_metadata,
                      "external_benign": paths.external_benign_metadata}
    expectations = base.build_feature_cache_expectations(
        panels=dict(panels),
        metadata_info={p: {"sha256": sha256_file(metadata_paths[p])} for p in PANEL_ORDER},
        metadata_paths=metadata_paths, metadata_root=paths.root,
        corpus_manifest_sha256=sha256_file(paths.manifest),
    )
    provider = base.ResumableEvaluationFeatureProvider(
        delegate=dual_eval.ForbiddenFrozenDelegate(contract), artifact=contract,
        expectations=expectations, cache_root=stage, rebuild=False, cache_only=True)
    entries, position = [], 0
    for panel in PANEL_ORDER:
        for row in panels[panel]:
            fused = np.concatenate((text[position], image[position])).astype(np.float64, copy=False)
            position += 1
            expected = expectations[row["sample_id"]]
            path = stage / panel / f"{row['sample_id']}.npz"
            provider._write(path, expected, fused)
            if not np.array_equal(provider._load(path, expected), fused):
                raise RuntimeError(f"cache round trip differs: {row['sample_id']}")
            entries.append({"sequence": position, "panel": panel,
                            "sample_id": row["sample_id"], "path": _portable(path),
                            "bytes": path.stat().st_size, "sha256": sha256_file(path),
                            "embedding_sha256": _embedding_hash(fused)})
    if position != 50 or len(list(stage.rglob("*.npz"))) != 50:
        raise RuntimeError("cache materialization did not produce exactly 50 entries")
    return entries


def _old_panel_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    heads = value.get("per_head_action_counts")
    if not isinstance(heads, Mapping):
        raise ValueError("generic panel summary has no per-head counts")
    return {
        "rows": value["rows"], "malicious_rows": value["malicious_rows"],
        "benign_rows": value["benign_rows"],
        "malicious_blocked": value["malicious_blocked"],
        "benign_blocked": value["benign_blocked"],
        "combined_action_counts": value["combined_action_counts"],
        "image_head_action_counts": heads["image"],
        "text_head_action_counts": heads["text"],
    }


def _compat_gates(panels: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    checks = {
        "external_benign_exact_counts": panels["external_benign"]["benign_rows"] == 20,
        "external_benign_fewer_than_3_blocked": panels["external_benign"]["benign_blocked"] < 3,
        "fixed_benign_fewer_than_3_blocked": panels["regression"]["benign_blocked"] < 3,
        "fixed_exact_counts": (panels["regression"]["malicious_rows"] == 10
                               and panels["regression"]["benign_rows"] == 10),
        "fixed_malicious_10_of_10_blocked": panels["regression"]["malicious_blocked"] == 10,
        "text_led_exact_counts": panels["text_led"]["malicious_rows"] == 10,
        "text_led_malicious_10_of_10_blocked": panels["text_led"]["malicious_blocked"] == 10,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _artifact_result(pair: Any, name: str) -> dict[str, Any]:
    artifact = pair.image_artifact if name == "image" else pair.text_artifact
    path = pair.paths.image if name == "image" else pair.paths.text
    return {
        "artifact_version": 1, "path": _portable(path), "sha256": sha256_file(path),
        "feature_dim": int(artifact.feature_dim), "pooling": artifact.pooling,
        "block_threshold": float(artifact.threshold),
        "review_threshold": float(pair.manifest["artifacts"][name]["review_threshold"]),
        "uncertainty_margin": float(artifact.uncertainty_margin),
        **dict(pair.manifest["shared_provenance"]),
    }


def _source_bindings(generic_dir: Path, source_dir: Path,
                      materialization: Path, declaration: Path):
    paths = {
        **{name: generic_dir / filename for name, filename in GENERIC_OUTPUT_NAMES.items()},
        **{name: source_dir / filename for name, filename in SOURCE_INPUT_NAMES.items()},
        "cache_materialization": materialization,
        "declaration_manifest": declaration,
        "implementation_manifest": declaration.parent / "implementation_manifest.json",
        "input_snapshot_manifest": declaration.parent / "input_snapshot_manifest.json",
        "model_snapshot_manifest": declaration.parent / "model_snapshot_manifest.json",
        "runtime_environment": declaration.parent / "runtime_environment.json",
        "strict_corpus_validation": declaration.parent / "strict_corpus_validation.json",
        "subject_claim": declaration.parent / "subject_claim.json",
    }
    return {name: _hash_binding(path) for name, path in paths.items()}


def _package_evidence(*, target: str, workspace: Path, stage: Path,
                      protocol: dict[str, Any], attempt: dict[str, Any], pair: Any,
                      corpus_report: dict[str, object], generic_dir: Path,
                      source_dir: Path, materialization_path: Path,
                      cache_entries: list[dict[str, Any]]) -> None:
    generic_json_path = generic_dir / GENERIC_OUTPUT_NAMES["generic_results_json"]
    generic_csv_path = generic_dir / GENERIC_OUTPUT_NAMES["generic_results_csv"]
    generic_validation_path = generic_dir / GENERIC_OUTPUT_NAMES["generic_validation_report"]
    generic = _json(generic_json_path, "generic evaluation")
    validated = _json(generic_validation_path, "generic validation")
    if (generic.get("schema_version") != 3 or generic.get("artifact_mode") != "dual_or"
            or generic.get("target") != target
            or generic.get("acceptance", {}).get("passed") is not True
            or validated.get("schema_version") != 1 or validated.get("passed") is not True
            or validated.get("target") != target
            or validated.get("pair_identity_sha256") != pair.manifest["pair_identity_sha256"]
            or validated.get("runtime_detector_identity_sha256")
            != pair.manifest["runtime_detector_identity_sha256"]):
        raise ValueError("generic evaluator/validator evidence did not pass exactly")
    validation_evidence = validated.get("evidence")
    if (not isinstance(validation_evidence, Mapping)
            or validation_evidence.get("evaluation_json_sha256") != sha256_file(generic_json_path)
            or validation_evidence.get("evaluation_csv_sha256") != sha256_file(generic_csv_path)):
        raise ValueError("generic validator does not bind generic outputs")
    dual_eval.validate_exact_frozen_cache_report(
        generic["runtime_validation"]["feature_cache"])
    sources = _source_bindings(generic_dir, source_dir, materialization_path,
                               workspace / "declaration_manifest.json")
    generic_rows = generic.get("results")
    if not isinstance(generic_rows, list) or len(generic_rows) != 50:
        raise ValueError("generic evaluation must contain exactly 50 rows")
    rows = []
    for source_row in generic_rows:
        if not isinstance(source_row, Mapping):
            raise ValueError("generic evaluation row is invalid")
        row = {name: source_row[name] for name in COMPAT_CSV_FIELDS}
        row["expected_action"] = "block" if int(row["label_id"]) == 1 else "allow"
        row["accepted"] = row["combined_action"] == row["expected_action"]
        rows.append(row)
    panels = {panel: _old_panel_summary(generic["panels"][panel])
              for panel in PANEL_ORDER}
    gates = _compat_gates(panels)
    if gates["passed"] is not True:
        raise ValueError("generic results fail frozen compatibility gates")
    by_id = {(entry["panel"], entry["sample_id"]): entry for entry in cache_entries}
    if len(by_id) != 50:
        raise ValueError("cache materialization contains duplicate identities")
    compat_entries, digest_lines = [], []
    for row in rows:
        entry = by_id.get((row["panel"], row["sample_id"]))
        if not isinstance(entry, Mapping) or row["cache_sha256"] != entry["sha256"]:
            raise ValueError("generic result/cache materialization differs")
        compat = {"panel": row["panel"], "sample_id": row["sample_id"],
                  "path": row["cache_path"], "bytes": entry["bytes"],
                  "sha256": entry["sha256"]}
        compat_entries.append(compat)
        digest_lines.append(f"{compat['path']}|{compat['bytes']}|{compat['sha256']}")
    cache_tree = hashlib.sha256(("\n".join(digest_lines) + "\n").encode()).hexdigest()
    results = {
        "schema_version": 1, "package_format": GENERIC_DERIVED_PACKAGE,
        "qualification_execution": protocol["qualification_execution"],
        "qualification_subject_sha256": protocol["qualification_subject_sha256"],
        "promotable": protocol["promotable"],
        "evaluation_id": protocol["protocol_id"], "generated_utc": generic["generated_utc"],
        "promotion_authorized": False, "one_shot_attempt": attempt,
        "protocol": {"path": _portable(workspace / "protocol.json"),
                     "sha256": sha256_file(workspace / "protocol.json"),
                     "payload": protocol},
        "generic_sources": sources,
        "corpus_manifest": {"path": protocol["corpus_binding"]["manifest"],
                            "sha256": protocol["corpus_binding"]["manifest_sha256"],
                            "schema_version": 4},
        "corpus_validation": corpus_report, "metadata": generic["metadata"],
        "artifacts": {
            "development_manifest": _hash_binding(workspace / "development_manifest.json"),
            "shared_provenance": dict(pair.manifest["shared_provenance"]),
            "image_head": _artifact_result(pair, "image"),
            "text_head": _artifact_result(pair, "text"),
        },
        "split_contract": {"fused_dimension": pair.base_feature_dim * 2,
                           "text_slice": [0, pair.base_feature_dim],
                           "image_slice": [pair.base_feature_dim, pair.base_feature_dim * 2]},
        "action_contract": dict(ACTION_CONTRACT), "panels": panels, "gates": gates,
        "cache": {"report": generic["runtime_validation"]["feature_cache"],
                  "delegate_object_calls": 0, "entries": compat_entries,
                  "entry_tree_sha256": cache_tree},
        "results": rows,
    }
    stage.mkdir(parents=True, exist_ok=True)
    results_json = stage / "dual_head_frozen_results.json"
    results_csv = stage / "dual_head_frozen_results.csv"
    base.atomic_write_json(results_json, results)
    _write_csv(results_csv, rows, COMPAT_CSV_FIELDS)
    attempt_path = stage / "attempt_started.json"
    _copy(workspace / "consume_started.json", attempt_path)
    evaluation_manifest = {
        "schema_version": 1, "package_format": GENERIC_DERIVED_PACKAGE,
        "qualification_execution": protocol["qualification_execution"],
        "qualification_subject_sha256": protocol["qualification_subject_sha256"],
        "promotable": protocol["promotable"],
        "evaluation_id": protocol["protocol_id"], "row_count": 50,
        "corpus_manifest_sha256": protocol["corpus_binding"]["manifest_sha256"],
        "cache_entry_tree_sha256": cache_tree,
        "artifacts": {"image_head_sha256": sha256_file(pair.paths.image),
                      "text_head_sha256": sha256_file(pair.paths.text)},
        "development_manifest": _hash_binding(workspace / "development_manifest.json"),
        "protocol": _hash_binding(workspace / "protocol.json"),
        "attempt": _hash_binding(attempt_path),
        "evaluator": _hash_binding(workspace / "evaluate_once.py"),
        "validator": _hash_binding(workspace / "validate_once.py"),
        "panel_metadata_sha256": {
            "regression": protocol["corpus_binding"]["fixed_metadata_sha256"],
            "text_led": protocol["corpus_binding"]["text_led_metadata_sha256"],
            "external_benign": protocol["corpus_binding"]["external_benign_metadata_sha256"],
        },
        "generic_sources": sources,
        "outputs": {"dual_head_frozen_results.json": _hash_binding(results_json),
                    "dual_head_frozen_results.csv": _hash_binding(results_csv)},
    }
    evaluation_manifest_path = stage / "evaluation_manifest.json"
    base.atomic_write_json(evaluation_manifest_path, evaluation_manifest)
    validation_report = {
        "schema_version": 1, "package_format": GENERIC_DERIVED_PACKAGE,
        "qualification_execution": protocol["qualification_execution"],
        "qualification_subject_sha256": protocol["qualification_subject_sha256"],
        "promotable": protocol["promotable"],
        "validation_id": f"{protocol['protocol_id']}-validator-v1",
        "generated_utc": validated["generated_utc"], "ok": True,
        "promotion_authorized": False,
        "evaluation_manifest": _hash_binding(evaluation_manifest_path),
        "generic_sources": sources, "gates": gates, "panels": panels,
        "evidence": {
            "image_artifact_sha256": sha256_file(pair.paths.image),
            "text_artifact_sha256": sha256_file(pair.paths.text),
            "corpus_manifest_sha256": protocol["corpus_binding"]["manifest_sha256"],
            "development_manifest_sha256": sha256_file(workspace / "development_manifest.json"),
            "protocol_sha256": sha256_file(workspace / "protocol.json"),
            "cache_entry_tree_sha256": cache_tree,
            "results_json": _hash_binding(results_json),
            "results_csv": _hash_binding(results_csv),
        },
        "recomputation": {
            "rows": 50, "unique_sample_ids": 50,
            "all_per_head_scores_and_actions_recomputed": True,
            "all_combined_actions_recomputed": True,
            "csv_json_exact_copy_validated": True,
            "validator_cache": generic["runtime_validation"]["feature_cache"],
        },
    }
    validation_report_path = stage / "validation_report.json"
    base.atomic_write_json(validation_report_path, validation_report)
    base.atomic_write_json(stage / "validation_manifest.json", {
        "schema_version": 1, "package_format": GENERIC_DERIVED_PACKAGE,
        "qualification_execution": protocol["qualification_execution"],
        "qualification_subject_sha256": protocol["qualification_subject_sha256"],
        "promotable": protocol["promotable"],
        "promotion_authorized": False,
        "evaluation_manifest": _hash_binding(evaluation_manifest_path),
        "validation_report": _hash_binding(validation_report_path),
        "validator": _hash_binding(workspace / "validate_once.py"),
        "generic_sources": sources,
    })


def consume_workspace(
    *, target: str, run_root: Path, qualification_subject_sha256: str,
    development_workspace: Path | None = None,
) -> dict[str, Any]:
    """Consume one declaration; any failure permanently consumes its attempt."""
    paths = RunPaths(run_root)
    root = _workspace_for_subject(
        paths, target, qualification_subject_sha256, development_workspace
    )
    if not root.is_dir():
        raise FileNotFoundError(f"declared qualification workspace is missing: {root}")
    for path in (root / "extracted_features", root / "cache", root / "evidence"):
        if path.exists():
            raise FileExistsError(f"refusing to reuse qualification output: {path}")
    stale = [path for pattern in (".cache-staging-*", ".evidence-staging-*",
                                  ".generic-staging-*") for path in root.glob(pattern)]
    if stale:
        raise FileExistsError(f"stale qualification staging output exists: {stale}")
    (protocol, frozen_paths, panels, pair, corpus_report,
     _, model_before) = _validate_declaration(
        target=target, paths=paths, workspace=root,
        subject=qualification_subject_sha256,
    )
    if ((development_workspace is None) is not (protocol.get("promotable") is True)):
        raise ValueError("development workspace/promotable declaration mode differs")
    selected_summary = (
        frozen_paths.training_directory / target / "dual_or" / "training_summary.json"
    )
    attempt = {
        "schema_version": 1, "package_format": GENERIC_DERIVED_PACKAGE,
        "protocol_id": protocol["protocol_id"],
        "qualification_subject_sha256": qualification_subject_sha256,
        "protocol_sha256": sha256_file(root / "protocol.json"),
        "extractor_sha256": sha256_file(root / "extract_once.py"),
        "evaluator_sha256": sha256_file(root / "evaluate_once.py"),
        "implementation_manifest_sha256": sha256_file(root / "implementation_manifest.json"),
        "input_snapshot_manifest_sha256": sha256_file(root / "input_snapshot_manifest.json"),
        "model_snapshot_manifest_sha256": sha256_file(root / "model_snapshot_manifest.json"),
        "subject_claim_sha256": sha256_file(root / "subject_claim.json"),
        "qualification_execution": protocol["qualification_execution"],
        "execution_ordinal": 1, "retries_permitted": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    attempt_path = root / "consume_started.json"
    _exclusive_json(attempt_path, attempt)
    suffix = f"{os.getpid()}"
    cache_stage = root / f".cache-staging-{suffix}"
    evidence_stage = root / f".evidence-staging-{suffix}"
    generic_stage = root / f".generic-staging-{suffix}"
    try:
        runtime_contract = protocol.get("runtime_environment_contract")
        if not isinstance(runtime_contract, Mapping):
            raise ValueError("declared runtime environment contract is missing")
        runtime_environment_path = root / "runtime_environment.json"
        environment = _runtime_environment(
            target=target, protocol=protocol,
            runtime_image_id=runtime_contract.get("container_image_supplied_id"),
            model_before=model_before,
        )
        base.atomic_write_json(runtime_environment_path, environment)
        source_features = _run_declared_extractor(
            target=target, root=root, paths=frozen_paths, protocol=protocol,
        )
        model_after = _verify_model_snapshot_binding(
            protocol["model_snapshot_binding"]
        )
        if model_after != model_before:
            raise ValueError("declared model snapshot changed during extraction")
        environment["model_snapshot_after_extraction"] = model_after
        environment["model_snapshot_pre_post_equal"] = True
        base.atomic_write_json(runtime_environment_path, environment)
        text, image, feature_sources, feature_manifest = _load_bundle(
            source_features, protocol)
        cache_stage.mkdir()
        entries = _materialize_cache(
            cache_stage, panels, pair, frozen_paths, text, image
        )
        cache_root = root / "cache"
        os.replace(cache_stage, cache_root)
        _fsync_directory(cache_root.parent)
        entries = [
            {
                **entry,
                "path": _cache_lineage_path(
                    protocol=protocol,
                    subject=qualification_subject_sha256,
                    panel=entry["panel"],
                    sample_id=entry["sample_id"],
                ),
            }
            for entry in entries
        ]

        generic_stage.mkdir()
        _run_snapshot(root, "evaluate_bordair_dual_detector.py", (
            target, "--run-root", str(frozen_paths.root),
            "--summary", str(selected_summary),
            "--output-dir", str(generic_stage), "--cache-root", str(cache_root),
            "--qualification-workspace", str(root),
        ))
        generic_json = generic_stage / f"{target}_dual_or_results.json"
        generic_csv = generic_stage / f"{target}_dual_or_results.csv"
        generic_validation = generic_stage / f"{target}_dual_or_validation.json"
        _run_snapshot(root, "validate_bordair_dual_evaluation.py", (
            target, "--run-root", str(frozen_paths.root),
            "--evaluation-dir", str(generic_stage),
            "--training-dir", str(frozen_paths.training_directory),
            "--cache-root", str(cache_root), "--output", str(generic_validation),
            "--qualification-workspace", str(root),
        ))

        evidence_stage.mkdir()
        packaged_generic = evidence_stage / "generic"
        packaged_source = evidence_stage / "source_inputs"
        packaged_generic.mkdir()
        packaged_source.mkdir()
        for source, destination in (
            (generic_json, packaged_generic / GENERIC_OUTPUT_NAMES["generic_results_json"]),
            (generic_csv, packaged_generic / GENERIC_OUTPUT_NAMES["generic_results_csv"]),
            (generic_validation, packaged_generic / GENERIC_OUTPUT_NAMES["generic_validation_report"]),
            (feature_sources["feature_bundle"], packaged_source / SOURCE_INPUT_NAMES["feature_bundle"]),
            (feature_sources["feature_manifest"], packaged_source / SOURCE_INPUT_NAMES["feature_manifest"]),
            (feature_sources["aligned_source_metadata"], packaged_source / SOURCE_INPUT_NAMES["aligned_source_metadata"]),
            (selected_summary, packaged_source / SOURCE_INPUT_NAMES["training_summary"]),
            (pair.manifest_path, packaged_source / SOURCE_INPUT_NAMES["source_pair_manifest"]),
        ):
            _copy(source, destination)
        materialization_path = evidence_stage / "cache_materialization.json"
        materialization = {
            "schema_version": 1, "package_format": GENERIC_DERIVED_PACKAGE,
            "target": target, "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(root / "protocol.json"),
            "pair_identity_sha256": pair.manifest["pair_identity_sha256"],
            "runtime_detector_identity_sha256": pair.manifest["runtime_detector_identity_sha256"],
            "qualification_execution": protocol["qualification_execution"],
            "qualification_subject_sha256": qualification_subject_sha256,
            "runtime_environment": _hash_binding(runtime_environment_path),
            "model_snapshot_manifest": _hash_binding(root / "model_snapshot_manifest.json"),
            "input_snapshot_manifest": _hash_binding(root / "input_snapshot_manifest.json"),
            "subject_claim": _hash_binding(root / "subject_claim.json"),
            "source_feature_manifest": _hash_binding(packaged_source / SOURCE_INPUT_NAMES["feature_manifest"]),
            "source_feature_bundle": _hash_binding(packaged_source / SOURCE_INPUT_NAMES["feature_bundle"]),
            "source_aligned_metadata": _hash_binding(packaged_source / SOURCE_INPUT_NAMES["aligned_source_metadata"]),
            "source_manifest_payload_sha256": base.sha256_text(
                json.dumps(feature_manifest, sort_keys=True, separators=(",", ":"))),
            "counters": {"input_rows": 50, "unique_sample_ids": 50,
                         "duplicates": 0, "entries_written": 50,
                         "invalid_entries": 0, "partial_entries": 0,
                         "overwritten_entries": 0, "reused_entries": 0},
            "cache_contract": protocol["cache_contract"], "entries": entries,
        }
        base.atomic_write_json(materialization_path, materialization)
        _package_evidence(
            target=target, workspace=root, stage=evidence_stage, protocol=protocol,
            attempt=attempt, pair=pair, corpus_report=corpus_report,
            generic_dir=packaged_generic, source_dir=packaged_source,
            materialization_path=materialization_path, cache_entries=entries)
        final_evidence = root / "evidence"
        os.replace(evidence_stage, final_evidence)
        _fsync_directory(final_evidence.parent)
        return {
            "target": target, "workspace": _portable(root),
            "qualification_subject_sha256": qualification_subject_sha256,
            "promotable": protocol["promotable"],
            "protocol_id": protocol["protocol_id"],
            "attempt_sha256": sha256_file(attempt_path), "cache_entries": 50,
            "generic_evaluation": _portable(final_evidence / "generic" /
                                             GENERIC_OUTPUT_NAMES["generic_results_json"]),
            "generic_validation": _portable(final_evidence / "generic" /
                                             GENERIC_OUTPUT_NAMES["generic_validation_report"]),
            "compatibility_evidence": _portable(final_evidence),
            "state": "consumed_and_validated",
        }
    finally:
        for temporary in (cache_stage, evidence_stage, generic_stage):
            if temporary.exists():
                shutil.rmtree(temporary)


def generic_derived_paths(evidence_root: Path) -> dict[str, Path]:
    """Return every additional file required by a generic-derived package."""
    root = evidence_root.expanduser().resolve()
    package = root.parent
    paths = {
        "declaration_manifest": package / "declaration_manifest.json",
        "implementation_manifest": package / "implementation_manifest.json",
        "input_snapshot_manifest": package / "input_snapshot_manifest.json",
        "model_snapshot_manifest": package / "model_snapshot_manifest.json",
        "runtime_environment": package / "runtime_environment.json",
        "strict_corpus_validation": package / "strict_corpus_validation.json",
        "subject_claim": package / "subject_claim.json",
        "ordered_metadata": package / "metadata" / "ordered_frozen_metadata.csv",
        "cache_materialization": root / "cache_materialization.json",
        **{name: root / "generic" / filename
           for name, filename in GENERIC_OUTPUT_NAMES.items()},
        **{name: root / "source_inputs" / filename
           for name, filename in SOURCE_INPUT_NAMES.items()},
    }
    implementation_path = package / "implementation_manifest.json"
    if implementation_path.is_file():
        implementation = _json(implementation_path, "implementation manifest")
        files = implementation.get("files")
        if isinstance(files, Mapping):
            for relative in files:
                paths[f"implementation_snapshot::{relative}"] = (
                    package / "code_snapshot" / Path(relative)
                )
    input_path = package / "input_snapshot_manifest.json"
    if input_path.is_file():
        input_manifest = _json(input_path, "input snapshot manifest")
        files = input_manifest.get("files")
        if isinstance(files, Mapping):
            for relative in files:
                paths[f"input_snapshot::{relative}"] = (
                    package / "input_snapshot" / Path(relative)
                )
    return paths


def _binding_matches(binding: object, path: Path) -> bool:
    return (isinstance(binding, Mapping) and path.is_file()
            and binding.get("sha256") == sha256_file(path)
            and binding.get("bytes") == path.stat().st_size)


def _cache_lineage_path(
    *, protocol: Mapping[str, Any], subject: str, panel: str, sample_id: str,
) -> str:
    subject_payload = protocol.get("qualification_subject")
    version = (
        subject_payload.get("protocol_version")
        if isinstance(subject_payload, Mapping)
        else None
    )
    if version != QUALIFICATION_PROTOCOL_VERSION:
        raise ValueError("cache lineage protocol version is unsupported")
    return qualification_cache_locator(subject, panel, sample_id)


def validate_generic_derived_evidence(
    *, evidence_root: Path, protocol: Mapping[str, Any], results: Mapping[str, Any],
    report: Mapping[str, Any], evaluation_manifest: Mapping[str, Any],
    validation_manifest: Mapping[str, Any], compatibility_rows: list[Mapping[str, Any]],
    compatibility_panels: Mapping[str, Any], compatibility_gates: Mapping[str, Any],
    expected_image_sha256: str, expected_text_sha256: str,
    expected_corpus_sha256: str, expected_runtime_identity: str,
) -> dict[str, Any]:
    """Validate generic sources and their deterministic compatibility transform.

    This deliberately does not score or select anything.  It verifies the exact
    schema-3 evaluator output, independently generated validator report, extracted
    primitives, cache-materialization lineage, and byte-for-byte transform into the
    already independently checked schema-1 compatibility envelope.
    """
    evidence_root = evidence_root.expanduser().resolve()
    root = evidence_root.parent
    paths = generic_derived_paths(evidence_root)
    if not all(path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise FileNotFoundError(f"generic-derived evidence is incomplete: {missing}")
    if any(value.get("package_format") != GENERIC_DERIVED_PACKAGE for value in
           (protocol, results, report, evaluation_manifest, validation_manifest)):
        raise ValueError("generic-derived package marker differs")
    if (protocol.get("qualification_execution") != QUALIFICATION_EXECUTION
            or protocol.get("promotable") is not True
            or any(value.get("qualification_execution") != QUALIFICATION_EXECUTION
                   or value.get("promotable") is not True
                   for value in (results, report, evaluation_manifest,
                                 validation_manifest))):
        raise ValueError("generic-derived package is not atomic/promotable")
    subject = protocol.get("qualification_subject_sha256")
    if (not isinstance(subject, str) or len(subject) != 64
            or any(value.get("qualification_subject_sha256") != subject
                   for value in (results, report, evaluation_manifest,
                                 validation_manifest))):
        raise ValueError("generic-derived qualification subject differs")

    approved_scripts = Path(__file__).resolve().parents[1] / "scripts"
    if (sha256_file(root / "evaluate_once.py")
            != sha256_file(approved_scripts / "evaluate_bordair_dual_detector.py")
            or sha256_file(root / "validate_once.py")
            != sha256_file(approved_scripts / "validate_bordair_dual_evaluation.py")
            or sha256_file(root / "extract_once.py")
            != sha256_file(approved_scripts / "extract_mllm_features.py")):
        raise ValueError("generic-derived extractor/evaluator/validator is not approved")
    _validate_implementation_snapshot(
        root, tracked_match=True, current_runtime_match=False,
    )
    frozen_paths, _ = _validate_input_snapshot(root, target=str(protocol["target"]))
    if frozen_paths.root != (root / "input_snapshot" / "run").resolve():
        raise ValueError("generic-derived input snapshot root differs")

    declaration = _json(paths["declaration_manifest"], "declaration manifest")
    declared_files = declaration.get("files")
    declared_actual = {
        "protocol": root / "protocol.json", "evaluator": root / "evaluate_once.py",
        "validator": root / "validate_once.py",
        "extractor": root / "extract_once.py",
        "implementation_manifest": root / "implementation_manifest.json",
        "input_snapshot_manifest": root / "input_snapshot_manifest.json",
        "model_snapshot_manifest": root / "model_snapshot_manifest.json",
        "strict_corpus_validation": root / "strict_corpus_validation.json",
        "subject_claim": root / "subject_claim.json",
        "development_manifest": root / "development_manifest.json",
        "ordered_metadata": paths["ordered_metadata"],
        **{f"{panel}_metadata": root / "metadata" / METADATA_NAMES[panel]
           for panel in PANEL_ORDER},
    }
    if (declaration.get("schema_version") != 1
            or declaration.get("package_format") != GENERIC_DERIVED_PACKAGE
            or declaration.get("state") != "declared"
            or declaration.get("target") != protocol.get("target")
            or declaration.get("protocol_id") != protocol.get("protocol_id")
            or declaration.get("qualification_execution") != QUALIFICATION_EXECUTION
            or declaration.get("promotable") is not True
            or declaration.get("qualification_subject_sha256") != subject
            or declaration.get("protocol_sha256") != sha256_file(root / "protocol.json")
            or declaration.get("ordered_rows") != 50
            or declaration.get("ordered_unique_sample_ids") != 50
            or declaration.get("no_frozen_scores_observed") is not True
            or not isinstance(declared_files, Mapping)
            or set(declared_files) != set(declared_actual)
            or any(not _binding_matches(declared_files[name], path)
                   for name, path in declared_actual.items())):
        raise ValueError("generic-derived declaration binding differs")

    claim = _json(paths["subject_claim"], "subject claim receipt")
    implementation_manifest = _json(
        paths["implementation_manifest"], "implementation manifest"
    )
    model_manifest = _json(paths["model_snapshot_manifest"], "model snapshot manifest")
    runtime = _json(paths["runtime_environment"], "runtime environment")
    before = runtime.get("model_snapshot_before_extraction")
    after = runtime.get("model_snapshot_after_extraction")
    runtime_contract = protocol.get("runtime_environment_contract")
    container = runtime.get("container_image")
    packages = runtime.get("packages")
    required_packages = {
        "numpy", "torch", "transformers", "Pillow", "accelerate",
        "qwen-vl-utils", "safetensors",
    }
    if (claim.get("qualification_subject_sha256") != subject
            or claim.get("qualification_subject") != protocol.get("qualification_subject")
            or claim.get("target") != protocol.get("target")
            or claim.get("promotable") is not True
            or not _binding_matches(protocol.get("subject_claim"), paths["subject_claim"])
            or protocol.get("model_snapshot_binding") != model_manifest
            or runtime.get("qualification_execution") != QUALIFICATION_EXECUTION
            or runtime.get("qualification_subject_sha256") != subject
            or runtime.get("target") != protocol.get("target")
            or runtime.get("protocol_id") != protocol.get("protocol_id")
            or runtime.get("closure_scope")
            != "bound code plus recorded runtime environment"
            or runtime.get("model_snapshot_pre_post_equal") is not True
            or before != after
            or not isinstance(runtime_contract, Mapping)
            or not isinstance(container, Mapping)
            or container.get("supplied_id")
            != runtime_contract.get("container_image_supplied_id")
            or container.get("verification")
            != runtime_contract.get("container_image_verification")
            or not isinstance(packages, Mapping)
            or set(packages) != required_packages
            or not isinstance(runtime.get("python"), Mapping)
            or runtime.get("python", {}).get("version")
            != implementation_manifest.get("python_version")
            or packages.get("numpy") != implementation_manifest.get("numpy_version")
            or not isinstance(runtime.get("torch_runtime"), Mapping)):
        raise ValueError("generic-derived claim/model/runtime closure differs")
    expected_observation = {
        "schema_version": 1,
        "snapshots": {
            role: {
                "snapshot_path": entry["snapshot_path"],
                "content_sha256": entry["fingerprint"]["content_sha256"],
                "n_files": entry["fingerprint"]["n_files"],
                "total_bytes": entry["fingerprint"]["total_bytes"],
            }
            for role, entry in model_manifest.get("snapshots", {}).items()
        },
    }
    if before != expected_observation:
        raise ValueError("generic-derived model pre/post observations differ from manifest")

    actual_source_paths = {
        **{name: paths[name] for name in GENERIC_OUTPUT_NAMES},
        **{name: paths[name] for name in SOURCE_INPUT_NAMES},
        "cache_materialization": paths["cache_materialization"],
        "declaration_manifest": paths["declaration_manifest"],
        "implementation_manifest": paths["implementation_manifest"],
        "input_snapshot_manifest": paths["input_snapshot_manifest"],
        "model_snapshot_manifest": paths["model_snapshot_manifest"],
        "runtime_environment": paths["runtime_environment"],
        "strict_corpus_validation": paths["strict_corpus_validation"],
        "subject_claim": paths["subject_claim"],
    }
    source_maps = (results.get("generic_sources"),
                   evaluation_manifest.get("generic_sources"),
                   report.get("generic_sources"),
                   validation_manifest.get("generic_sources"))
    if (any(not isinstance(value, Mapping) for value in source_maps)
            or any(dict(value) != dict(source_maps[0]) for value in source_maps[1:])
            or set(source_maps[0]) != set(actual_source_paths)
            or any(not _binding_matches(source_maps[0][name], path)
                   for name, path in actual_source_paths.items())):
        raise ValueError("generic source binding differs or was substituted")

    generic = _json(paths["generic_results_json"], "generic results")
    generic_validation = _json(paths["generic_validation_report"], "generic validation")
    generic_rows = generic.get("results")
    if (generic.get("schema_version") != 3 or generic.get("evaluation_version") != "v7"
            or generic.get("artifact_mode") != "dual_or"
            or generic.get("target") != protocol.get("target")
            or not isinstance(generic_rows, list) or len(generic_rows) != 50
            or generic.get("acceptance", {}).get("passed") is not True):
        raise ValueError("generic evaluation envelope differs")
    snapshot_receipt = generic.get("runtime_validation", {}).get(
        "qualification_snapshot"
    )
    if (not isinstance(snapshot_receipt, Mapping)
            or snapshot_receipt.get("mode") != "declared_input_snapshot"
            or snapshot_receipt.get("protocol_sha256")
            != sha256_file(root / "protocol.json")
            or snapshot_receipt.get("declaration_manifest_sha256")
            != sha256_file(paths["declaration_manifest"])
            or snapshot_receipt.get("input_snapshot_manifest_sha256")
            != sha256_file(paths["input_snapshot_manifest"])
            or snapshot_receipt.get("qualification_subject_sha256") != subject
            or snapshot_receipt.get("qualification_execution")
            != QUALIFICATION_EXECUTION
            or snapshot_receipt.get("promotable") is not True):
        raise ValueError("generic evaluation qualification snapshot receipt differs")
    dual_validate.validate_csv_copy(paths["generic_results_csv"], generic_rows)
    pair_binding = generic.get("detector_pair")
    development = protocol.get("development_binding")
    if (not isinstance(pair_binding, Mapping) or not isinstance(development, Mapping)
            or pair_binding.get("manifest_sha256") != sha256_file(paths["source_pair_manifest"])
            or pair_binding.get("pair_identity_sha256") != development.get("pair_identity_sha256")
            or pair_binding.get("runtime_detector_identity_sha256") != expected_runtime_identity
            or pair_binding.get("composition") != "or"
            or pair_binding.get("cache_ordering") != ["text_tokens", "image_tokens"]
            or pair_binding.get("artifacts", {}).get("image", {}).get("sha256")
            != expected_image_sha256
            or pair_binding.get("artifacts", {}).get("text", {}).get("sha256")
            != expected_text_sha256):
        raise ValueError("generic evaluation detector-pair binding differs")
    source_pair = _json(paths["source_pair_manifest"], "source pair manifest")
    from .bordair_dual import validate_pair_manifest_payload
    validate_pair_manifest_payload(source_pair)
    if (source_pair.get("qualification_evidence") is not None
            or source_pair.get("pair_identity_sha256") != development.get("pair_identity_sha256")
            or source_pair.get("runtime_detector_identity_sha256") != expected_runtime_identity):
        raise ValueError("generic source pair is not the declared unqualified pair")
    source_summary = _json(paths["training_summary"], "source training summary")
    summary_pair = source_summary.get("artifact_pair")
    if (not isinstance(summary_pair, Mapping)
            or summary_pair.get("manifest_sha256") != sha256_file(paths["source_pair_manifest"])
            or summary_pair.get("pair_identity_sha256") != source_pair["pair_identity_sha256"]
            or summary_pair.get("runtime_detector_identity_sha256") != expected_runtime_identity
            or generic.get("training_summary", {}).get("sha256")
            != sha256_file(paths["training_summary"])):
        raise ValueError("generic source training/pair binding differs")

    evidence = generic_validation.get("evidence")
    if (generic_validation.get("schema_version") != 1
            or generic_validation.get("validation_kind") != "independent_dual_or_evaluation"
            or generic_validation.get("target") != protocol.get("target")
            or generic_validation.get("passed") is not True
            or generic_validation.get("pair_identity_sha256") != source_pair["pair_identity_sha256"]
            or generic_validation.get("runtime_detector_identity_sha256") != expected_runtime_identity
            or not isinstance(evidence, Mapping)
            or evidence.get("evaluation_json_sha256") != sha256_file(paths["generic_results_json"])
            or evidence.get("evaluation_csv_sha256") != sha256_file(paths["generic_results_csv"])
            or evidence.get("training_summary_sha256") != sha256_file(paths["training_summary"])
            or evidence.get("pair_manifest_sha256") != sha256_file(paths["source_pair_manifest"])
            or evidence.get("image_artifact_sha256") != expected_image_sha256
            or evidence.get("text_artifact_sha256") != expected_text_sha256
            or evidence.get("corpus_manifest_sha256") != expected_corpus_sha256
            or evidence.get("cache_entries") != 50
            or evidence.get("cache_statistics") != ONE_SHOT_CACHE_STATISTICS):
        raise ValueError("generic independent validation binding differs")
    dual_eval.validate_exact_frozen_cache_report(
        generic["runtime_validation"]["feature_cache"])

    expected_compat = []
    for row in generic_rows:
        transformed = {name: row[name] for name in COMPAT_CSV_FIELDS}
        transformed["expected_action"] = "block" if int(transformed["label_id"]) == 1 else "allow"
        transformed["accepted"] = transformed["combined_action"] == transformed["expected_action"]
        expected_compat.append(transformed)
    if expected_compat != compatibility_rows:
        raise ValueError("generic results/compatibility transform diverges")
    expected_panels = {panel: _old_panel_summary(generic["panels"][panel])
                       for panel in PANEL_ORDER}
    if (expected_panels != compatibility_panels
            or _compat_gates(expected_panels) != compatibility_gates):
        raise ValueError("generic panels/gates compatibility transform diverges")

    text, image, _, feature_manifest = _load_bundle(
        paths["feature_bundle"].parent, protocol,
        declared_metadata_override=paths["ordered_metadata"])
    materialization = _json(paths["cache_materialization"], "cache materialization")
    counters = {"input_rows": 50, "unique_sample_ids": 50, "duplicates": 0,
                "entries_written": 50, "invalid_entries": 0, "partial_entries": 0,
                "overwritten_entries": 0, "reused_entries": 0}
    entries = materialization.get("entries")
    if (materialization.get("schema_version") != 1
            or materialization.get("package_format") != GENERIC_DERIVED_PACKAGE
            or materialization.get("qualification_execution")
            != QUALIFICATION_EXECUTION
            or materialization.get("qualification_subject_sha256") != subject
            or materialization.get("target") != protocol.get("target")
            or materialization.get("protocol_id") != protocol.get("protocol_id")
            or materialization.get("protocol_sha256") != sha256_file(root / "protocol.json")
            or materialization.get("pair_identity_sha256") != source_pair["pair_identity_sha256"]
            or materialization.get("runtime_detector_identity_sha256") != expected_runtime_identity
            or materialization.get("cache_contract") != protocol.get("cache_contract")
            or materialization.get("counters") != counters
            or not isinstance(entries, list) or len(entries) != 50
            or materialization.get("source_manifest_payload_sha256")
            != base.sha256_text(json.dumps(feature_manifest, sort_keys=True,
                                           separators=(",", ":")))):
        raise ValueError("cache materialization contract differs")
    for name, path in (
        ("runtime_environment", paths["runtime_environment"]),
        ("model_snapshot_manifest", paths["model_snapshot_manifest"]),
        ("input_snapshot_manifest", paths["input_snapshot_manifest"]),
        ("subject_claim", paths["subject_claim"]),
    ):
        if not _binding_matches(materialization.get(name), path):
            raise ValueError(f"cache materialization closure differs: {name}")
    for name, key in (("feature_manifest", "source_feature_manifest"),
                      ("feature_bundle", "source_feature_bundle"),
                      ("aligned_source_metadata", "source_aligned_metadata")):
        if not _binding_matches(materialization.get(key), paths[name]):
            raise ValueError(f"cache materialization source differs: {name}")
    cache_hashes: dict[str, str] = {}
    cases = protocol["ordered_cases"]
    for index, (entry, case, generic_row) in enumerate(zip(entries, cases, generic_rows)):
        fused = np.concatenate((text[index], image[index]))
        cache_locator = _cache_lineage_path(
            protocol=protocol,
            subject=subject,
            panel=case["panel"],
            sample_id=case["sample_id"],
        )
        copied_cache = root / "cache" / case["panel"] / f"{case['sample_id']}.npz"
        if (not isinstance(entry, Mapping) or entry.get("sequence") != index + 1
                or entry.get("panel") != case["panel"]
                or entry.get("sample_id") != case["sample_id"]
                or entry.get("embedding_sha256") != _embedding_hash(fused)
                or entry.get("sha256") != generic_row["cache_sha256"]
                or entry.get("path") != cache_locator
                or generic_row.get("cache_path") != cache_locator
                or not copied_cache.is_file()
                or copied_cache.is_symlink()
                or copied_cache.stat().st_nlink != 1
                or entry.get("sha256") != sha256_file(copied_cache)
                or entry.get("bytes") != copied_cache.stat().st_size
                or isinstance(entry.get("bytes"), bool)
                or not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0):
            raise ValueError("cache materialization row differs")
        cache_hashes[f"{case['panel']}/{case['sample_id']}.npz"] = entry["sha256"]
    generic_cache_tree = base.sha256_text(
        json.dumps(cache_hashes, sort_keys=True, separators=(",", ":")))
    if evidence.get("cache_tree_sha256") != generic_cache_tree:
        raise ValueError("generic validator cache tree differs from materialization")
    return {
        "source_format": GENERIC_DERIVED_PACKAGE,
        "generic_evaluation_sha256": sha256_file(paths["generic_results_json"]),
        "generic_validation_sha256": sha256_file(paths["generic_validation_report"]),
        "source_pair_identity_sha256": source_pair["pair_identity_sha256"],
        "source_feature_bundle_sha256": sha256_file(paths["feature_bundle"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Declare or consume one exact Bordair v7 dual qualification.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("declare", "consume"):
        command = commands.add_parser(name)
        command.add_argument("target", choices=tuple(base.EXPECTED_TARGETS))
        command.add_argument("--run-root", type=Path, default=base.RUN)
        command.add_argument(
            "--development-workspace", type=Path,
            help=(
                "Explicit synthetic/test workspace. Evidence produced with this "
                "option is marked non-promotable."
            ),
        )
        if name == "declare":
            command.add_argument("--model-cache-dir", type=Path,
                                 default=Path("models/huggingface"))
            command.add_argument("--llava-runtime-model", type=Path)
            command.add_argument(
                "--runtime-image-id",
                default=os.environ.get("AEGIS_QUALIFICATION_IMAGE_ID"),
                help="Caller-inspected container image ID; recorded as unverified input.",
            )
        else:
            command.add_argument(
                "--qualification-subject", required=True,
                help="SHA-256 returned by the one successful declare command.",
            )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "declare":
        result = declare_workspace(
            target=args.target, run_root=args.run_root,
            model_cache_dir=args.model_cache_dir,
            llava_runtime_model=args.llava_runtime_model,
            runtime_image_id=args.runtime_image_id,
            development_workspace=args.development_workspace,
        )
    else:
        result = consume_workspace(
            target=args.target, run_root=args.run_root,
            qualification_subject_sha256=args.qualification_subject,
            development_workspace=args.development_workspace,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
