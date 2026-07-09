from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MSSBenchImportConfig:
    source_name: str = "MSSBench"
    source_split: str = "test"
    include_chat: bool = True
    include_embodied: bool = True


def import_mssbench_json(
    input_json: str | Path,
    output_csv: str | Path,
    config: MSSBenchImportConfig | None = None,
) -> pd.DataFrame:
    """Normalize local MSSBench combined.json into AEGIS metadata."""
    config = config or MSSBenchImportConfig()
    input_path = Path(input_json)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows: list[dict[str, Any]] = []
    if config.include_chat:
        for index, item in enumerate(payload.get("chat", [])):
            rows.extend(_chat_rows(item, index, config))
    if config.include_embodied:
        for index, item in enumerate(payload.get("embodied", [])):
            rows.extend(_embodied_rows(item, index, config))

    if not rows:
        raise ValueError(f"No MSSBench records were imported from {input_json}.")

    table = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def _chat_rows(
    item: dict[str, Any],
    index: int,
    config: MSSBenchImportConfig,
) -> list[dict[str, Any]]:
    rows = []
    queries = item.get("queries") or []
    for query_index, query in enumerate(queries):
        base = {
            "source": config.source_name,
            "split": config.source_split,
            "task_type": "chat",
            "source_index": index,
            "query_index": query_index,
            "source_id": f"mssbench_chat_{index:06d}_q{query_index:02d}",
            "harm_category": _clean_category(item.get("Type")),
            "source_category": item.get("Type", ""),
            "intent": item.get("intent", ""),
            "unsafe_context_description": item.get("unsafe_image", ""),
            "template_id": "mssbench_chat",
            "modality": "multimodal",
            "text": str(query),
        }
        rows.append(
            _row(
                base,
                sample_id=f"mssbench_chat_{index:06d}_q{query_index:02d}_safe",
                label="benign",
                source_subset="chat_safe",
                image_context="safe",
                image_path=f"chat/{item.get('safe_image_path', '')}",
                attack_style="none",
                notes="Safe image context for a situational-safety query.",
            )
        )
        rows.append(
            _row(
                base,
                sample_id=f"mssbench_chat_{index:06d}_q{query_index:02d}_unsafe",
                label="malicious",
                source_subset="chat_unsafe",
                image_context="unsafe",
                image_path=f"chat/{item.get('unsafe_image_path', '')}",
                attack_style="situational_unsafe_context",
                notes="Unsafe image context makes the query safety-risky.",
            )
        )
    return rows


def _embodied_rows(
    item: dict[str, Any],
    index: int,
    config: MSSBenchImportConfig,
) -> list[dict[str, Any]]:
    rows = []
    for instruction_index, instruction in enumerate(item.get("safe_instructions") or []):
        base = _embodied_base(item, index, instruction_index, config, instruction)
        rows.append(
            _row(
                base,
                sample_id=f"mssbench_embodied_{index:06d}_i{instruction_index:02d}_safe",
                label="benign",
                source_subset="embodied_safe",
                image_context="safe",
                image_path=f"embodied/{item.get('safe', '')}",
                attack_style="none",
                notes="Safe embodied context for an instruction.",
            )
        )

    for instruction_index, instruction in enumerate(item.get("unsafe_instructions") or []):
        base = _embodied_base(item, index, instruction_index, config, instruction)
        rows.append(
            _row(
                base,
                sample_id=f"mssbench_embodied_{index:06d}_i{instruction_index:02d}_unsafe",
                label="malicious",
                source_subset="embodied_unsafe",
                image_context="unsafe",
                image_path=f"embodied/{item.get('unsafe', '')}",
                attack_style="situational_unsafe_context",
                notes="Unsafe embodied context makes the instruction safety-risky.",
            )
        )
    return rows


def _embodied_base(
    item: dict[str, Any],
    index: int,
    instruction_index: int,
    config: MSSBenchImportConfig,
    instruction: str,
) -> dict[str, Any]:
    return {
        "source": config.source_name,
        "split": config.source_split,
        "task_type": "embodied",
        "source_index": index,
        "query_index": instruction_index,
        "source_id": f"mssbench_embodied_{index:06d}_i{instruction_index:02d}",
        "harm_category": _clean_category(item.get("category")),
        "source_category": item.get("category", ""),
        "intent": item.get("task", ""),
        "unsafe_context_description": item.get("observation_unsafe", ""),
        "safe_context_description": item.get("observation_safe", ""),
        "template_id": "mssbench_embodied",
        "modality": "multimodal",
        "text": str(instruction),
    }


def _row(
    base: dict[str, Any],
    sample_id: str,
    label: str,
    source_subset: str,
    image_context: str,
    image_path: str,
    attack_style: str,
    notes: str,
) -> dict[str, Any]:
    row = dict(base)
    row.update(
        {
            "sample_id": sample_id,
            "label": label,
            "source_subset": source_subset,
            "image_context": image_context,
            "image_path": image_path.replace("\\", "/"),
            "attack_style": attack_style,
            "notes": notes,
        }
    )
    return row


def _clean_category(value: Any) -> str:
    if value is None:
        return "situational_safety"
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return text or "situational_safety"
