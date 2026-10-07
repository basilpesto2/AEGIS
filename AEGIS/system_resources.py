from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Callable

from AEGIS.provenance import is_sha256, model_directory_fingerprint


@dataclass(frozen=True)
class ResourceRequirements:
    min_total_physical_bytes: int = 0
    min_available_physical_bytes: int = 0
    min_available_virtual_bytes: int = 0
    min_disk_free_bytes: int = 0
    min_cuda_device_memory_bytes: int = 0
    min_model_cache_bytes: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"resources.{name} must be a non-negative integer.")


@dataclass(frozen=True)
class ResourceSnapshot:
    total_physical_bytes: int | None
    available_physical_bytes: int | None
    total_virtual_bytes: int | None
    available_virtual_bytes: int | None
    disk_free_bytes: int | None
    model_cache_bytes: int | None
    cuda_devices: tuple[dict[str, object], ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def inspect_system_resources(
    *,
    disk_path: str | Path,
    model_path: str | Path | None = None,
    cuda_probe: Callable[[], tuple[dict[str, object], ...]] | None = None,
) -> ResourceSnapshot:
    warnings: list[str] = []
    memory = _memory_snapshot(warnings)
    disk_free = _disk_free_bytes(Path(disk_path), warnings)
    model_bytes = None
    if model_path is not None:
        path = Path(model_path)
        if path.exists():
            try:
                model_bytes = _path_size(path)
            except OSError as exc:
                warnings.append(f"Could not measure model cache size: {exc}")
    cuda_devices: tuple[dict[str, object], ...] = ()
    if cuda_probe is not None:
        try:
            cuda_devices = tuple(cuda_probe())
        except Exception as exc:
            warnings.append(f"CUDA resource probe failed: {type(exc).__name__}: {exc}")
    return ResourceSnapshot(
        total_physical_bytes=memory[0],
        available_physical_bytes=memory[1],
        total_virtual_bytes=memory[2],
        available_virtual_bytes=memory[3],
        disk_free_bytes=disk_free,
        model_cache_bytes=model_bytes,
        cuda_devices=cuda_devices,
        warnings=tuple(warnings),
    )


def evaluate_resource_requirements(
    requirements: ResourceRequirements,
    snapshot: ResourceSnapshot,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def minimum(name: str, actual: int | None, required: int) -> None:
        if required <= 0:
            return
        checks.append(
            {
                "name": name,
                "ok": actual is not None and actual >= required,
                "actual_bytes": actual,
                "required_bytes": required,
                "detail": f"actual={_format_bytes(actual)}, required>={_format_bytes(required)}",
            }
        )

    minimum(
        "total_physical_memory",
        snapshot.total_physical_bytes,
        requirements.min_total_physical_bytes,
    )
    minimum(
        "available_physical_memory",
        snapshot.available_physical_bytes,
        requirements.min_available_physical_bytes,
    )
    minimum(
        "available_virtual_memory",
        snapshot.available_virtual_bytes,
        requirements.min_available_virtual_bytes,
    )
    minimum("disk_free", snapshot.disk_free_bytes, requirements.min_disk_free_bytes)
    minimum(
        "model_cache_size",
        snapshot.model_cache_bytes,
        requirements.min_model_cache_bytes,
    )
    if requirements.min_cuda_device_memory_bytes > 0:
        largest_available = max(
            (
                int(device.get("free_memory_bytes", 0))
                for device in snapshot.cuda_devices
            ),
            default=0,
        )
        minimum(
            "cuda_device_available_memory",
            largest_available,
            requirements.min_cuda_device_memory_bytes,
        )
    failed_names = {str(check["name"]) for check in checks if not check["ok"]}
    recommendations = []
    if "total_physical_memory" in failed_names:
        recommendations.append("Use a host with more installed RAM or select a smaller target profile.")
    if "available_physical_memory" in failed_names:
        recommendations.append("Close memory-heavy applications before starting the model worker.")
    if "available_virtual_memory" in failed_names:
        recommendations.append("Increase the Windows paging file or Linux swap/commit limit.")
    if "disk_free" in failed_names:
        recommendations.append("Free disk space or move the model cache to a larger volume.")
    if "model_cache_size" in failed_names:
        recommendations.append("Complete the target model download before offline startup.")
    if "cuda_device_available_memory" in failed_names:
        recommendations.append(
            "Free GPU memory, use a GPU with more VRAM, or select a separately "
            "validated smaller/quantized target."
        )
    return {
        "ok": all(bool(check["ok"]) for check in checks),
        "requirements": asdict(requirements),
        "snapshot": snapshot.to_dict(),
        "checks": checks,
        "recommendations": recommendations,
    }


def resolve_model_storage_path(
    *,
    model_id: str | None,
    cache_dir: str | Path | None,
    root: str | Path = ".",
) -> Path | None:
    if not model_id:
        return None
    model_path = Path(model_id)
    if not model_path.is_absolute():
        model_path = Path(root) / model_path
    if model_path.exists():
        return model_path.resolve()
    if cache_dir is None or "/" not in model_id:
        return None
    cache_path = Path(cache_dir)
    repo_dir = "models--" + model_id.replace("/", "--")
    candidate = cache_path / repo_dir
    return candidate.resolve() if candidate.exists() else None


def inspect_model_cache(
    model_path: str | Path | None,
    *,
    model_family: str,
    revision: str | None = None,
    expected_content_sha256: str | None = None,
) -> dict[str, object]:
    if model_path is None:
        return {
            "ok": False,
            "kind": "missing",
            "path": None,
            "revision": revision or None,
            "snapshot_path": None,
            "checks": [],
            "missing_files": ["model cache directory"],
            "recommendations": [
                "Download the complete target checkpoint at the configured revision."
            ],
        }
    root = Path(model_path).resolve()
    if not root.is_dir():
        return {
            "ok": False,
            "kind": "missing",
            "path": str(root),
            "revision": revision or None,
            "snapshot_path": None,
            "checks": [],
            "missing_files": [str(root)],
            "recommendations": [
                "Repair the configured model path or download the checkpoint again."
            ],
        }

    snapshot, kind, resolved_revision = _model_snapshot(root, revision)
    checks: list[dict[str, object]] = []
    missing_files: list[str] = []

    def required_file(name: str) -> None:
        path = snapshot / name
        ok = _is_nonempty_file(path)
        checks.append({"name": name, "ok": ok, "path": str(path)})
        if not ok:
            missing_files.append(name)

    required_file("config.json")
    tokenizer_candidates = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    tokenizer_ok = any(
        _is_nonempty_file(snapshot / name)
        for name in tokenizer_candidates
    )
    checks.append(
        {
            "name": "tokenizer_files",
            "ok": tokenizer_ok,
            "candidates": list(tokenizer_candidates),
        }
    )
    if not tokenizer_ok:
        missing_files.append("one tokenizer file")
    if model_family in {"qwen25_vl", "llava_onevision"}:
        required_file("preprocessor_config.json")

    weight_files = sorted(
        path
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in snapshot.glob(pattern)
        if _is_nonempty_file(path)
    )
    checks.append(
        {
            "name": "model_weights",
            "ok": bool(weight_files),
            "files": [path.name for path in weight_files],
        }
    )
    if not weight_files:
        missing_files.append("model weight files")

    index_candidates = sorted(snapshot.glob("*.index.json"))
    missing_shards: list[str] = []
    invalid_indexes: list[str] = []
    for index_path in index_candidates:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map", {})
            if not isinstance(weight_map, dict) or not weight_map:
                invalid_indexes.append(index_path.name)
                continue
            for shard in sorted({str(value) for value in weight_map.values()}):
                shard_path = snapshot / shard
                if not _is_nonempty_file(shard_path):
                    missing_shards.append(shard)
        except (AttributeError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            invalid_indexes.append(index_path.name)
    checks.append(
        {
            "name": "weight_indexes_complete",
            "ok": not invalid_indexes and not missing_shards,
            "indexes": [path.name for path in index_candidates],
            "invalid_indexes": invalid_indexes,
            "missing_shards": missing_shards,
        }
    )
    missing_files.extend(missing_shards)
    missing_files.extend(invalid_indexes)

    revision_ok = not revision or kind != "huggingface_cache" or resolved_revision == revision
    checks.append(
        {
            "name": "requested_revision_available",
            "ok": revision_ok,
            "requested": revision or None,
            "resolved": resolved_revision,
        }
    )
    if not revision_ok:
        missing_files.append(f"snapshot revision {revision}")

    expected_content = expected_content_sha256
    if expected_content is None and kind == "local_directory" and revision and is_sha256(revision):
        expected_content = revision
    if expected_content is not None:
        if not is_sha256(expected_content):
            raise ValueError("expected_content_sha256 must be a SHA-256 hex digest.")
        try:
            content_sha256 = str(model_directory_fingerprint(snapshot)["content_sha256"])
            content_ok = content_sha256 == expected_content
        except (OSError, ValueError):
            content_sha256 = None
            content_ok = False
        checks.append(
            {
                "name": "model_runtime_content_sha256",
                "ok": content_ok,
                "expected": expected_content,
                "computed": content_sha256,
            }
        )
        if not content_ok:
            missing_files.append("model runtime content matching configured SHA-256")

    ok = all(bool(check["ok"]) for check in checks)
    return {
        "ok": ok,
        "kind": kind,
        "path": str(root),
        "revision": resolved_revision,
        "expected_content_sha256": expected_content,
        "snapshot_path": str(snapshot),
        "checks": checks,
        "missing_files": sorted(set(missing_files)),
        "recommendations": []
        if ok
        else [
            "Repair or re-download the complete checkpoint at the configured revision."
        ],
    }


def _model_snapshot(
    root: Path,
    revision: str | None,
) -> tuple[Path, str, str | None]:
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return root, "local_directory", revision or None
    if revision:
        return snapshots / revision, "huggingface_cache", revision
    reference = root / "refs" / "main"
    if reference.is_file():
        resolved = reference.read_text(encoding="utf-8").strip()
        if resolved:
            return snapshots / resolved, "huggingface_cache", resolved
    available = sorted(path for path in snapshots.iterdir() if path.is_dir())
    if len(available) == 1:
        return available[0], "huggingface_cache", available[0].name
    return snapshots / "__missing_revision__", "huggingface_cache", None


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _memory_snapshot(warnings: list[str]) -> tuple[int | None, ...]:
    if sys.platform == "win32":
        try:
            return _windows_memory_snapshot()
        except (OSError, AttributeError) as exc:
            warnings.append(f"Windows memory probe failed: {exc}")
            return (None, None, None, None)
    total_physical = available_physical = None
    total_virtual = available_virtual = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            values = _parse_meminfo(meminfo.read_text(encoding="utf-8"))
            total_physical = values.get("MemTotal")
            available_physical = values.get("MemAvailable")
            total_virtual = values.get("CommitLimit")
            committed = values.get("Committed_AS")
            if total_virtual is not None and committed is not None:
                available_virtual = max(0, total_virtual - committed)
        except OSError as exc:
            warnings.append(f"Linux memory probe failed: {exc}")
    if total_physical is None or available_physical is None:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_physical = total_physical or (
                page_size * int(os.sysconf("SC_PHYS_PAGES"))
            )
            available_physical = available_physical or (
                page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
            )
        except (AttributeError, OSError, ValueError) as exc:
            warnings.append(f"Physical memory fallback probe failed: {exc}")
    return total_physical, available_physical, total_virtual, available_virtual


def _windows_memory_snapshot() -> tuple[int, int, int, int]:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return (
        int(status.ullTotalPhys),
        int(status.ullAvailPhys),
        int(status.ullTotalPageFile),
        int(status.ullAvailPageFile),
    )


def _disk_free_bytes(path: Path, warnings: list[str]) -> int | None:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    try:
        return int(shutil.disk_usage(candidate).free)
    except OSError as exc:
        warnings.append(f"Disk probe failed: {exc}")
        return None


def _path_size(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    return sum(
        int(item.stat().st_size)
        for item in path.rglob("*")
        if item.is_file()
    )


def _parse_meminfo(content: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in content.splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        fields = raw.strip().split()
        if not fields:
            continue
        multiplier = 1024 if len(fields) > 1 and fields[1].lower() == "kb" else 1
        try:
            values[name] = int(fields[0]) * multiplier
        except ValueError:
            continue
    return values


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    gib = value / (1024**3)
    return f"{value} bytes ({gib:.2f} GiB)"
