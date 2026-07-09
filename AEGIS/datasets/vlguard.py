from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class VLGuardImportConfig:
    source_split: str
    source_name: str = "VLGuard"
    include_safe_safes: bool = True
    include_safe_unsafes: bool = True
    include_unsafes: bool = True


def import_vlguard_json(
    input_json: str | Path,
    output_csv: str | Path,
    config: VLGuardImportConfig,
) -> pd.DataFrame:
    """Normalize a local VLGuard JSON file into AEGIS metadata.

    VLGuard schema, based on the public evaluation/conversion scripts:
    - each row has `image`, `safe`, and `instr-resp`
    - unsafe image rows use `instr-resp[0]["instruction"]`
    - safe image rows can include `safe_instruction` and `unsafe_instruction`
    """
    with Path(input_json).open("r", encoding="utf-8") as handle:
        raw_rows = json.load(handle)

    records: list[dict] = []
    for index, item in enumerate(raw_rows):
        image_path = str(item.get("image", "")).strip()
        is_safe_image = bool(item.get("safe", False))
        instr_resp = item.get("instr-resp", [])
        if not isinstance(instr_resp, list):
            raise ValueError(f"Row {index} has invalid 'instr-resp' field.")

        if not is_safe_image and config.include_unsafes:
            instruction = _first_value(instr_resp, "instruction")
            if instruction:
                records.append(
                    asdict(
                        _record(
                            item=item,
                            index=index,
                            subset="unsafes",
                            label="malicious",
                            text=instruction,
                            image_path=image_path,
                            config=config,
                            attack_style="unsafe_image_context",
                            harm_category="situational_unsafe",
                        )
                    )
                )

        if is_safe_image and config.include_safe_safes:
            instruction = _first_value(instr_resp, "safe_instruction")
            if instruction:
                records.append(
                    asdict(
                        _record(
                            item=item,
                            index=index,
                            subset="safe_safes",
                            label="benign",
                            text=instruction,
                            image_path=image_path,
                            config=config,
                            attack_style="none",
                            harm_category="none",
                        )
                    )
                )

        if is_safe_image and config.include_safe_unsafes:
            instruction = _first_value(instr_resp, "unsafe_instruction")
            if instruction:
                records.append(
                    asdict(
                        _record(
                            item=item,
                            index=index,
                            subset="safe_unsafes",
                            label="malicious",
                            text=instruction,
                            image_path=image_path,
                            config=config,
                            attack_style="unsafe_text_on_safe_image",
                            harm_category="harmful_instruction",
                        )
                    )
                )

    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError(f"No VLGuard records were imported from {input_json}.")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


@dataclass(frozen=True)
class _AEGISRecord:
    sample_id: str
    label: str
    harm_category: str
    attack_style: str
    modality: str
    split: str
    text: str
    image_path: str
    source: str
    template_id: str
    source_subset: str
    source_index: int
    source_id: str
    source_harmful_category: str
    source_harmful_subcategory: str
    notes: str = ""


def _record(
    item: dict,
    index: int,
    subset: str,
    label: str,
    text: str,
    image_path: str,
    config: VLGuardImportConfig,
    attack_style: str,
    harm_category: str,
) -> _AEGISRecord:
    return _AEGISRecord(
        sample_id=f"vlguard_{config.source_split}_{subset}_{index:06d}",
        label=label,
        harm_category=harm_category,
        attack_style=attack_style,
        modality="image_text",
        split=config.source_split,
        text=text,
        image_path=image_path,
        source=config.source_name,
        template_id=f"vlguard_{subset}",
        source_subset=subset,
        source_index=index,
        source_id=str(item.get("id", "")),
        source_harmful_category=str(item.get("harmful_category", "")),
        source_harmful_subcategory=str(item.get("harmful_subcategory", "")),
    )


def _first_value(rows: list[dict], key: str) -> str | None:
    for row in rows:
        if isinstance(row, dict) and key in row and row[key] is not None:
            value = str(row[key]).strip()
            if value:
                return value
    return None
