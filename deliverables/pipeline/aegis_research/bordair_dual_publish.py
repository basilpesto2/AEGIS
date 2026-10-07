from __future__ import annotations

"""Fail-closed publication of an already-qualified dual detector package."""

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from . import bordair_evaluation as base
from .bordair_dual import load_detector_pair, sha256_file
from .bordair_dual_promotion import (
    validate_dual_promotion_candidate,
    validate_dual_source_candidate,
)
from .bordair_paths import REPOSITORY, RunPaths


FailureHook = Callable[[str], None]
REPARSE_POINT = 0x400


class PublicationValidationError(RuntimeError):
    """A new package was installed, failed validation, and was quarantined."""

    def __init__(self, quarantine: Path, cause: BaseException) -> None:
        self.state = "published_but_quarantined"
        self.quarantine = quarantine
        super().__init__(
            f"published_but_quarantined: {quarantine}; validation error: {cause}"
        )


class PublicationQuarantineError(RuntimeError):
    """Validation and quarantine both failed; canonical bytes and lock are retained."""

    def __init__(
        self, canonical: Path, validation_error: BaseException, quarantine_error: BaseException
    ) -> None:
        self.state = "published_validation_failed_quarantine_failed"
        self.canonical = canonical
        self.validation_error = validation_error
        self.quarantine_error = quarantine_error
        super().__init__(
            f"published_validation_failed_quarantine_failed: canonical={canonical}; "
            f"validation error: {validation_error}; quarantine error: {quarantine_error}"
        )


def _is_link_or_reparse(path: Path) -> bool:
    stat = path.lstat()
    return path.is_symlink() or bool(getattr(stat, "st_file_attributes", 0) & REPARSE_POINT)


def _fsync_directory(path: Path) -> None:
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


def _safe_relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a non-empty relative path")
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError(f"{label} path escapes the package")
    return Path(*pure.parts)


def package_files(pair: Any) -> dict[Path, tuple[Path, str]]:
    """Return every byte bound by the deployable pair, including qualification evidence."""

    files: dict[Path, tuple[Path, str]] = {
        Path("detector_pair_manifest.json"): (
            pair.manifest_path,
            sha256_file(pair.manifest_path),
        )
    }
    for name in ("image", "text"):
        entry = pair.manifest["artifacts"][name]
        relative = _safe_relative(entry["path"], f"{name} artifact")
        files[relative] = (pair.manifest_path.parent / relative, entry["sha256"])
    qualification = pair.manifest.get("qualification_evidence")
    if not isinstance(qualification, dict) or not isinstance(qualification.get("files"), dict):
        raise ValueError("qualified package has no bound qualification evidence tree")
    for name, entry in qualification["files"].items():
        relative = _safe_relative(entry.get("path"), f"qualification evidence {name}")
        if relative in files:
            raise ValueError(f"duplicate package path: {relative}")
        files[relative] = (pair.manifest_path.parent / relative, entry.get("sha256"))
    for relative, (source, expected) in files.items():
        if (
            not source.is_file()
            or _is_link_or_reparse(source)
            or source.stat().st_nlink != 1
            or sha256_file(source) != expected
        ):
            raise ValueError(f"source package file is missing or changed: {relative}")
    return files


def _candidate_runtime_preflight(
    config_path: Path, pair: Any, destination: Path
) -> None:
    """Reject a stale/single-head config before any destination mutation."""

    resolved = config_path.expanduser().resolve()
    raw = base.read_json(resolved)
    from AEGIS.deployment_config import load_deployment_config

    # Parse the complete production schema before mutation. This parser resolves
    # paths but does not load detector bytes; the latter is deliberately deferred
    # until the destination has been atomically installed.
    load_deployment_config(resolved)
    if raw.get("schema_version") != 2 or raw.get("target_profile") != pair.manifest["target"]:
        raise ValueError("candidate runtime config must be schema 2 for the pair target")
    policy = raw.get("policy")
    if not isinstance(policy, dict) or any(
        policy.get(name, "missing") is not None
        for name in ("block_threshold", "review_threshold", "review_margin")
    ):
        raise ValueError("candidate runtime config must use per-head thresholds")
    detector = raw.get("detector")
    heads = detector.get("heads") if isinstance(detector, dict) else None
    if detector.get("mode") != "or" or not isinstance(heads, list) or len(heads) != 2:
        raise ValueError("candidate runtime config must declare exactly two OR heads")
    base_dir = (resolved.parent / raw.get("base_dir", "")).resolve()
    configured = {head.get("name"): head for head in heads if isinstance(head, dict)}
    if set(configured) != {"image", "text"}:
        raise ValueError("candidate runtime config head set differs from pair")
    for name in ("image", "text"):
        entry = pair.manifest["artifacts"][name]
        expected_path = (destination / _safe_relative(entry["path"], name)).resolve()
        actual_path = (base_dir / str(configured[name].get("artifact", ""))).resolve()
        if actual_path != expected_path:
            raise ValueError(f"candidate runtime {name} path does not name the destination")
        if float(configured[name].get("review_threshold")) != float(entry["review_threshold"]):
            raise ValueError(f"candidate runtime {name} review threshold differs")
    options = raw.get("provider_options")
    shared = pair.manifest["shared_provenance"]
    if not isinstance(options, dict) or options.get("pooling") != "text_image_tokens":
        raise ValueError("candidate runtime config does not use fused pooling")
    if int(options.get("feature_dim", -1)) != int(shared["base_feature_dim"]) * 2:
        raise ValueError("candidate runtime feature dimension differs from pair")
    for field in ("model_id", "model_revision", "tokenizer_revision", "layer"):
        if options.get(field) != shared[field]:
            raise ValueError(f"candidate runtime provenance differs for {field}")
    from AEGIS.target_profiles import get_target_profile

    profile = get_target_profile(pair.manifest["target"])
    if profile.detector_mode != "or" or profile.provider != raw.get("provider"):
        raise ValueError("candidate target profile is not the pair's OR provider")
    if profile.provider_options != options:
        raise ValueError("candidate runtime provider options differ from target profile")
    profile_heads = {head.name: head for head in profile.detector_heads}
    if set(profile_heads) != {"image", "text"}:
        raise ValueError("candidate target profile remains single-head or incomplete")
    identity_entries: dict[str, dict[str, Any]] = {}
    for name in ("image", "text"):
        entry = pair.manifest["artifacts"][name]
        head = profile_heads[name]
        declared = Path(head.detector)
        profile_path = (
            declared.resolve()
            if declared.is_absolute()
            else (base_dir / declared).resolve()
        )
        expected_path = (destination / _safe_relative(entry["path"], name)).resolve()
        if profile_path != expected_path or head.detector_sha256 != entry["sha256"]:
            raise ValueError(f"candidate target profile {name} path/hash differs")
        if float(head.review_threshold) != float(entry["review_threshold"]):
            raise ValueError(f"candidate target profile {name} review threshold differs")
        identity_entries[name] = {
            "sha256": entry["sha256"],
            "review_threshold": float(entry["review_threshold"]),
        }
    from .bordair_dual import runtime_detector_identity_sha256

    if runtime_detector_identity_sha256(identity_entries) != pair.manifest["runtime_detector_identity_sha256"]:
        raise ValueError("candidate target profile identity differs from pair")


def _exact_destination(
    destination: Path, files: dict[Path, tuple[Path, str]]
) -> bool:
    if not destination.is_dir() or _is_link_or_reparse(destination):
        return False
    entries = list(destination.rglob("*"))
    if any(_is_link_or_reparse(path) for path in entries):
        return False
    actual_files = {path.relative_to(destination) for path in entries if path.is_file()}
    expected_dirs = {
        parent
        for relative in files
        for parent in relative.parents
        if str(parent) != "."
    }
    actual_dirs = {path.relative_to(destination) for path in entries if path.is_dir()}
    if actual_files != set(files) or actual_dirs != expected_dirs:
        return False
    return all(
        (destination / relative).stat().st_nlink == 1
        and sha256_file(destination / relative) == expected
        for relative, (_, expected) in files.items()
    )


def publish_pair_package(
    *,
    pair: Any,
    destination: Path,
    expected_target: str,
    post_validate: Callable[[Path, Path, Path], dict[str, Any]],
    failure_hook: FailureHook | None = None,
) -> dict[str, Any]:
    """Atomically publish one exact package; divergent destinations are immutable."""

    destination = destination.expanduser().resolve()
    if pair.manifest.get("target") != expected_target:
        raise ValueError("detector pair target differs from expected publication target")
    files = package_files(pair)
    lock = destination.parent / f".{destination.name}.publish.lock"
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage: Path | None = None
    created = False
    retain_lock = False
    try:
        os.close(lock_fd)
        if destination.exists():
            if not _exact_destination(destination, files):
                raise FileExistsError("destination exists but is not byte-identical")
        else:
            stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
            for relative, (source, expected) in files.items():
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                    dst.flush()
                    os.fsync(dst.fileno())
                if sha256_file(source) != expected or sha256_file(target) != expected:
                    raise ValueError(f"source changed while publishing: {relative}")
            load_detector_pair(stage / "detector_pair_manifest.json", expected_target=expected_target)
            for directory in sorted(
                (path for path in stage.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                _fsync_directory(directory)
            _fsync_directory(stage)
            if failure_hook:
                failure_hook("before_rename")
            _fsync_directory(destination.parent)
            os.rename(stage, destination)
            _fsync_directory(destination.parent)
            stage = None
            created = True
            if failure_hook:
                failure_hook("after_rename")
        if not _exact_destination(destination, files):
            raise ValueError("published destination changed before validation")
        entries = pair.manifest["artifacts"]
        result = post_validate(
            destination / "detector_pair_manifest.json",
            destination / _safe_relative(entries["image"]["path"], "image"),
            destination / _safe_relative(entries["text"]["path"], "text"),
        )
        if not isinstance(result, dict) or result.get("passed") is not True:
            raise ValueError("post-publication promotion validator did not pass")
        if not _exact_destination(destination, files):
            raise ValueError("published destination changed during validation")
        return {
            "schema_version": 1,
            "target": expected_target,
            "passed": True,
            "destination": base.display_path(destination),
            "published": created,
            "idempotent": not created,
            "pair_identity_sha256": pair.manifest["pair_identity_sha256"],
            "runtime_detector_identity_sha256": pair.manifest["runtime_detector_identity_sha256"],
            "files": {str(path).replace("\\", "/"): digest for path, (_, digest) in sorted(files.items(), key=lambda item: str(item[0]))},
            "promotion_validation": result,
        }
    except BaseException as error:
        if created and destination.exists():
            quarantine = destination.parent / (
                f".{destination.name}.published_but_quarantined-{uuid.uuid4().hex}"
            )
            try:
                os.rename(destination, quarantine)
                _fsync_directory(destination.parent)
            except BaseException as quarantine_error:
                retain_lock = True
                raise PublicationQuarantineError(
                    destination, error, quarantine_error
                ) from quarantine_error
            raise PublicationValidationError(quarantine, error) from error
        raise
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        if not retain_lock:
            lock.unlink(missing_ok=True)


def publish_qualified_dual_pair(
    *, target: str, run_paths: RunPaths, destination: Path, runtime_config: Path
) -> dict[str, Any]:
    training_dir = run_paths.training_directory
    manifest_path = training_dir / target / "dual_or" / "detector_pair_manifest.json"
    _, pair, _ = validate_dual_source_candidate(
        target=target,
        run_paths=run_paths,
        training_dir=training_dir,
        manifest_path=manifest_path,
    )
    destination = destination.expanduser().resolve()
    models_root = (REPOSITORY / "models" / "aegis").resolve()
    if destination.parent != models_root or destination == models_root:
        raise ValueError("destination must be one named directory directly under models/aegis")
    _candidate_runtime_preflight(runtime_config, pair, destination)

    def validate(pair_manifest: Path, image: Path, text: Path) -> dict[str, Any]:
        return validate_dual_promotion_candidate(
            target=target,
            run_paths=run_paths,
            evaluation_dir=run_paths.evaluation_directory,
            training_dir=training_dir,
            manifest_path=manifest_path,
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
