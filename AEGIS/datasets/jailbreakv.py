from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class JailBreakVImportConfig:
    source_name: str = "JailBreakV-28K"
    source_split: str = "test"
    image_root_prefix: str = "JailBreakV_28K"
    mini_only: bool = False


def import_jailbreakv_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    config: JailBreakVImportConfig | None = None,
) -> pd.DataFrame:
    """Normalize a local JailBreakV-28K CSV into AEGIS metadata.

    JailBreakV is malicious-only, so downstream experiments should pair it with
    benign controls from VLGuard, MSSBench, or a project-specific benign corpus.
    """
    config = config or JailBreakVImportConfig()
    raw = pd.read_csv(input_csv)
    _require_columns(raw, ["jailbreak_query", "image_path"])

    if config.mini_only:
        if "selected_mini" not in raw.columns:
            raise ValueError("--mini-only requires a 'selected_mini' column.")
        raw = raw[_as_bool_series(raw["selected_mini"])].reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for row_index, item in raw.reset_index(drop=True).iterrows():
        prompt = _as_string(item.get("jailbreak_query"))
        if not prompt:
            continue

        source_id = _as_string(item.get("id")) or str(row_index)
        image_path = _normalize_image_path(item.get("image_path"), config.image_root_prefix)
        attack_style = _slug(item.get("format"), default="jailbreak")
        harm_category = _slug(item.get("policy"), default="jailbreak")
        source_subset = _source_subset(image_path=image_path, fallback=item.get("format"))
        selected_mini = _as_bool(item.get("selected_mini"))
        transfer_from_llm = _as_bool(item.get("transfer_from_llm"))

        records.append(
            {
                "sample_id": f"jailbreakv_{config.source_split}_{_safe_id(source_id, row_index)}",
                "label": "malicious",
                "harm_category": harm_category,
                "attack_style": attack_style,
                "modality": "image_text" if image_path else "text",
                "split": config.source_split,
                "text": prompt,
                "image_path": image_path,
                "source": config.source_name,
                "template_id": f"jailbreakv_{attack_style}",
                "source_subset": source_subset,
                "source_index": row_index,
                "source_id": source_id,
                "base_query": _as_string(item.get("redteam_query")),
                "source_policy": _as_string(item.get("policy")),
                "source_format": _as_string(item.get("format")),
                "source_origin": _as_string(item.get("from")),
                "selected_mini": selected_mini,
                "transfer_from_llm": transfer_from_llm,
                "notes": "JailBreakV malicious jailbreak prompt; pair with benign controls before binary evaluation.",
            }
        )

    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError(f"No JailBreakV records were imported from {input_csv}.")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def _require_columns(table: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"JailBreakV CSV is missing required columns: {missing}")


def _as_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _as_bool_series(values: pd.Series) -> pd.Series:
    return values.map(_as_bool)


def _normalize_image_path(value: Any, prefix: str) -> str:
    image_path = _as_string(value).replace("\\", "/").lstrip("/")
    if not image_path:
        return ""
    normalized_prefix = prefix.strip("/").replace("\\", "/")
    if image_path == normalized_prefix or image_path.startswith(f"{normalized_prefix}/"):
        return image_path
    return f"{normalized_prefix}/{image_path}"


def _source_subset(image_path: str, fallback: Any) -> str:
    parts = [part for part in image_path.split("/") if part]
    if len(parts) >= 2:
        return _slug(parts[1], default="jailbreak")
    return _slug(fallback, default="jailbreak")


def _slug(value: Any, default: str) -> str:
    text = _as_string(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or default


def _safe_id(value: str, fallback: int) -> str:
    text = value.strip()
    if not text:
        return f"{fallback:06d}"
    try:
        return f"{int(float(text)):06d}"
    except ValueError:
        return _slug(text, default=f"{fallback:06d}")
