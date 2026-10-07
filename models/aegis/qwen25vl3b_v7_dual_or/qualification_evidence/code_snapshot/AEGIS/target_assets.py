from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TargetAsset:
    relative_path: str
    path: Path | None
    source: str | None

    @property
    def exists(self) -> bool:
        return self.path is not None and self.path.is_file()

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "path": None if self.path is None else str(self.path),
            "source": self.source,
            "exists": self.exists,
        }


def resolve_target_asset(
    relative_path: str,
    *,
    source_root: str | Path = ".",
) -> TargetAsset:
    path = Path(relative_path)
    candidate = path.resolve() if path.is_absolute() else (Path(source_root) / path).resolve()
    if candidate.is_file():
        return TargetAsset(relative_path, candidate, "source_checkout")
    return TargetAsset(relative_path, None, None)
