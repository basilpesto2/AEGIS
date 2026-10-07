from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def preprocessing_fingerprint(model_family: str, **settings: Any) -> str:
    """Return a stable fingerprint for embedding-affecting preprocessing settings."""

    payload = {
        "model_family": str(model_family),
        "settings": settings,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def model_directory_fingerprint(path: str | Path) -> dict[str, object]:
    supplied_path = Path(path)
    root = supplied_path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {root}")
    digest = hashlib.sha256()
    records: list[dict[str, object]] = []
    all_files = sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    included_files = [
        item for item in all_files if _include_transformers_runtime_file(item, root)
    ]
    excluded_paths = [
        item.relative_to(root).as_posix()
        for item in all_files
        if item not in included_files
    ]
    for file_path in included_files:
        relative = file_path.relative_to(root).as_posix()
        file_digest = _sha256_file(file_path)
        size = int(file_path.stat().st_size)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
        records.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": file_digest,
            }
        )
    if not records:
        raise ValueError(f"Model directory contains no files: {root}")
    return {
        "schema_version": 1,
        "fingerprint_profile": "transformers_pytorch_runtime_v1",
        "path": str(supplied_path),
        "content_sha256": digest.hexdigest(),
        "n_files": len(records),
        "total_bytes": sum(int(item["bytes"]) for item in records),
        "files": records,
        "excluded_paths": excluded_paths,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _include_transformers_runtime_file(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in {".cache", "onnx"} for part in relative.parts):
        return False
    if path.name == ".gitattributes" or path.suffix.lower() in {
        ".gif",
        ".jpeg",
        ".jpg",
        ".md",
        ".png",
    }:
        return False
    return True
