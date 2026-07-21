from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
from typing import Callable

from AEGIS.system_resources import inspect_model_cache
from AEGIS.target_profiles import TargetProfile, get_target_profile


MODEL_PREPARATION_SCHEMA_VERSION = 1


def prepare_target_model(
    target: str,
    *,
    root: str | Path = ".",
    download: bool = False,
    token_env: str = "HF_TOKEN",
    snapshot_download_fn: Callable[..., str] | None = None,
) -> dict[str, object]:
    """Plan, download, and verify the exact model source for one target profile."""

    profile = get_target_profile(target)
    root_path = Path(root).resolve()
    source = profile.model_source
    destination = _destination(profile, root_path)
    inspection_path = destination if destination.exists() else None
    before = inspect_model_cache(
        inspection_path,
        model_family=profile.model_family,
        revision=_integrity_revision(profile),
    )
    dependency_available = importlib.util.find_spec("huggingface_hub") is not None
    disk = _disk_preflight(destination, profile)
    token_configured = bool(os.getenv(token_env))
    performed = False

    if download and not bool(before["ok"]):
        if not disk["ok"]:
            raise RuntimeError(
                "Model preparation refused because disk preflight failed: "
                f"{disk['detail']}"
            )
        downloader = snapshot_download_fn or _snapshot_download()
        arguments: dict[str, object] = {
            "repo_id": source.repo_id,
            "revision": source.revision,
            "allow_patterns": list(source.allow_patterns),
        }
        token = os.getenv(token_env)
        if token:
            arguments["token"] = token
        if source.storage_mode == "local_directory":
            destination.parent.mkdir(parents=True, exist_ok=True)
            arguments["local_dir"] = str(destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            arguments["cache_dir"] = str(_cache_root(profile, root_path))
        downloader(**arguments)
        performed = True

    after_path = destination if destination.exists() else None
    after = inspect_model_cache(
        after_path,
        model_family=profile.model_family,
        revision=_integrity_revision(profile),
    )
    if download and not bool(after["ok"]):
        raise RuntimeError(
            "Downloaded checkpoint failed AEGIS integrity verification: "
            f"missing={after.get('missing_files', [])}"
        )

    ready = bool(after["ok"])
    preflight_ok = bool(disk["ok"])
    return {
        "schema_version": MODEL_PREPARATION_SCHEMA_VERSION,
        "ok": ready if download else preflight_ok,
        "ready": ready,
        "preflight_ok": preflight_ok,
        "status": (
            "ready"
            if ready
            else "download_required"
            if preflight_ok
            else "insufficient_disk"
        ),
        "target": profile.name,
        "model_family": profile.model_family,
        "source": {
            "repo_id": source.repo_id,
            "revision": source.revision,
            "storage_mode": source.storage_mode,
            "expected_content_sha256": source.expected_content_sha256,
            "allow_patterns": list(source.allow_patterns),
        },
        "destination": str(destination),
        "download_requested": bool(download),
        "download_performed": performed,
        "huggingface_hub_available": dependency_available,
        "token_env": token_env,
        "token_configured": token_configured,
        "disk_preflight": disk,
        "integrity": after,
        "next": (
            "docker compose up the matching AEGIS service"
            if ready
            else f"aegis prepare --target {profile.name} --download"
        ),
        "safety_note": (
            "Only the pinned target repository/revision and declared runtime file "
            "patterns are downloaded. Authentication token values are never reported."
        ),
    }


def _destination(profile: TargetProfile, root: Path) -> Path:
    source = profile.model_source
    if source.storage_mode == "local_directory":
        return (root / str(source.local_directory)).resolve()
    repo_dir = "models--" + source.repo_id.replace("/", "--")
    return (_cache_root(profile, root) / repo_dir).resolve()


def _cache_root(profile: TargetProfile, root: Path) -> Path:
    return (root / profile.cache_dir).resolve()


def _integrity_revision(profile: TargetProfile) -> str:
    source = profile.model_source
    if source.storage_mode == "local_directory":
        return str(source.expected_content_sha256 or "")
    return source.revision


def _disk_preflight(destination: Path, profile: TargetProfile) -> dict[str, object]:
    existing = destination
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    free = int(shutil.disk_usage(existing).free)
    required = int(profile.resources.get("min_disk_free_bytes", 0))
    return {
        "ok": free >= required,
        "path": str(existing),
        "available_bytes": free,
        "required_bytes": required,
        "detail": f"available={free}, required>={required}",
    }


def _snapshot_download() -> Callable[..., str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Model download requires the MLLM dependencies. Install with "
            "`pip install 'aegis-mllm-guard[mllm]'`."
        ) from exc
    return snapshot_download
