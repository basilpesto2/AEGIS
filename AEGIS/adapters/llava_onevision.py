from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image

from AEGIS.provenance import preprocessing_fingerprint


PoolingMode = Literal["last_token", "mean_tokens", "text_tokens", "image_tokens"]


@dataclass(frozen=True)
class LlavaOnevisionExtractionConfig:
    model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    runtime_model_id: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    cache_dir: str | None = "models/huggingface"
    layer: int = -1
    add_generation_prompt: bool = False
    torch_dtype: str = "auto"
    device_map: str = "auto"
    max_image_edge: int | None = 384
    batch_size: int = 1
    max_batch_characters: int | None = None
    max_samples: int | None = None
    local_files_only: bool = False


def extract_llava_onevision_pooling_embeddings(
    metadata_path: str | Path,
    corpus_root: str | Path,
    output_dir: str | Path,
    poolings: Sequence[PoolingMode],
    config: LlavaOnevisionExtractionConfig | None = None,
    output_prefix: str | None = None,
) -> dict[PoolingMode, Path]:
    """Extract several LLaVA-OneVision pooling views in one model pass."""
    config = config or LlavaOnevisionExtractionConfig()
    pooling_ids = _normalize_poolings(poolings)
    arrays, sample_ids = _extract_arrays(
        metadata_path=metadata_path,
        corpus_root=corpus_root,
        config=config,
        poolings=pooling_ids,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or Path(metadata_path).stem
    paths: dict[PoolingMode, Path] = {}
    for pooling in pooling_ids:
        output_path = output_root / f"{prefix}_layer_{layer_tag(config.layer)}_{pooling}.npz"
        np.savez(
            output_path,
            embeddings=np.vstack(arrays[pooling]).astype(np.float32),
            sample_id=np.asarray(sample_ids),
            model_id=np.asarray([config.model_id]),
            model_family=np.asarray(["llava_onevision"]),
            model_revision=np.asarray([config.model_revision or ""]),
            tokenizer_revision=np.asarray(
                [config.tokenizer_revision or config.model_revision or ""]
            ),
            preprocessing_sha256=np.asarray([llava_preprocessing_sha256(config)]),
            layer=np.asarray([config.layer]),
            pooling=np.asarray([pooling]),
        )
        paths[pooling] = output_path
    return paths


def _extract_arrays(
    metadata_path: str | Path,
    corpus_root: str | Path,
    config: LlavaOnevisionExtractionConfig,
    poolings: Sequence[PoolingMode],
) -> tuple[dict[PoolingMode, list[np.ndarray]], list[str]]:
    metadata = pd.read_csv(metadata_path)
    if "sample_id" not in metadata or "text" not in metadata:
        raise ValueError("Metadata must contain 'sample_id' and 'text' columns.")
    if "image_path" not in metadata:
        metadata["image_path"] = ""
    if config.max_samples is not None:
        metadata = metadata.head(config.max_samples).copy()

    torch, processor, model = _load_llava_runtime(
        config.runtime_model_id or config.model_id,
        config.cache_dir,
        config.torch_dtype,
        config.device_map,
        config.model_revision,
        config.tokenizer_revision,
        config.local_files_only,
    )
    image_token_id = int(model.config.image_token_index)
    root = Path(corpus_root)
    arrays: dict[PoolingMode, list[np.ndarray]] = {pooling: [] for pooling in poolings}
    sample_ids: list[str] = []
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    with torch.no_grad():
        rows = list(metadata.itertuples(index=False))
        for batch_rows in _make_batches(
            rows,
            batch_size=config.batch_size,
            max_batch_characters=config.max_batch_characters,
        ):
            images = []
            prompts = []
            for row in batch_rows:
                image_path = getattr(row, "image_path", "")
                if pd.isna(image_path) or not str(image_path).strip():
                    raise ValueError("LLaVA-OneVision extraction currently requires one image per row.")
                full_path = root / str(image_path)
                if not full_path.exists():
                    raise FileNotFoundError(f"Image path does not exist: {full_path}")
                with Image.open(full_path) as source_image:
                    image = source_image.convert("RGB")
                if config.max_image_edge is not None:
                    if config.max_image_edge <= 0:
                        raise ValueError("max_image_edge must be positive when set.")
                    image.thumbnail(
                        (config.max_image_edge, config.max_image_edge),
                        Image.Resampling.LANCZOS,
                    )
                images.append(image)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": str(getattr(row, "text"))},
                        ],
                    }
                ]
                prompts.append(
                    processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=config.add_generation_prompt,
                    )
                )
            inputs = processor(
                images=images,
                text=prompts,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
            outputs = model.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states.")
            hidden = hidden_states[config.layer]
            for batch_index, row in enumerate(batch_rows):
                for pooling in poolings:
                    arrays[pooling].append(
                        _pool_hidden_state(
                            hidden,
                            inputs.get("attention_mask"),
                            inputs.get("input_ids"),
                            image_token_id=image_token_id,
                            pooling=pooling,
                            batch_index=batch_index,
                        )
                    )
                sample_ids.append(str(getattr(row, "sample_id")))
    return arrays, sample_ids


@lru_cache(maxsize=4)
def _load_llava_runtime(
    model_id: str,
    cache_dir: str | None,
    torch_dtype: str,
    device_map: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    local_files_only: bool = False,
):
    """Load and retain one LLaVA runtime per model configuration."""
    torch, AutoProcessor, LlavaModel = _load_dependencies()
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        revision=tokenizer_revision or model_revision,
        local_files_only=local_files_only,
    )
    model = LlavaModel.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        revision=model_revision,
        local_files_only=local_files_only,
    )
    model.eval()
    return torch, processor, model


def llava_preprocessing_sha256(config: LlavaOnevisionExtractionConfig) -> str:
    return preprocessing_fingerprint(
        "llava_onevision",
        adapter_schema=1,
        add_generation_prompt=config.add_generation_prompt,
        max_image_edge=config.max_image_edge,
        padding=True,
        chat_template=True,
        image_mode="RGB",
    )


def clear_llava_runtime_cache() -> None:
    """Release cached LLaVA runtime references during shutdown or tests."""
    _load_llava_runtime.cache_clear()


def llava_runtime_cache_info() -> dict[str, int]:
    return dict(_load_llava_runtime.cache_info()._asdict())


def _make_batches(rows, batch_size: int, max_batch_characters: int | None):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_batch_characters is not None and max_batch_characters <= 0:
        raise ValueError("max_batch_characters must be positive when set.")
    batches = []
    current = []
    current_characters = 0
    for row in rows:
        row_characters = len(str(getattr(row, "text")))
        exceeds_rows = len(current) >= batch_size
        exceeds_characters = (
            max_batch_characters is not None
            and current
            and current_characters + row_characters > max_batch_characters
        )
        if exceeds_rows or exceeds_characters:
            batches.append(current)
            current = []
            current_characters = 0
        current.append(row)
        current_characters += row_characters
    if current:
        batches.append(current)
    return batches


def _pool_hidden_state(
    hidden_state,
    attention_mask,
    input_ids,
    image_token_id: int,
    pooling: PoolingMode,
    batch_index: int = 0,
) -> np.ndarray:
    if input_ids is None:
        raise ValueError("LLaVA pooling requires input_ids.")
    if hidden_state.shape[1] != input_ids.shape[1]:
        raise ValueError(
            "LLaVA hidden-state and input-token lengths differ; image-token pooling cannot be aligned safely. "
            f"Got {hidden_state.shape[1]} and {input_ids.shape[1]}."
        )
    mask = (
        hidden_state.new_ones(hidden_state.shape[1], dtype=bool)
        if attention_mask is None
        else attention_mask[batch_index].to(hidden_state.device).bool()
    )
    ids = input_ids[batch_index].to(hidden_state.device)
    if pooling == "last_token":
        selected = hidden_state[batch_index, mask.nonzero()[-1].item()]
    elif pooling == "mean_tokens":
        selected = _masked_mean(hidden_state, mask, pooling, batch_index)
    elif pooling == "image_tokens":
        selected = _masked_mean(
            hidden_state, mask & (ids == image_token_id), pooling, batch_index
        )
    elif pooling == "text_tokens":
        selected = _masked_mean(
            hidden_state, mask & (ids != image_token_id), pooling, batch_index
        )
    else:
        raise ValueError(f"Unsupported pooling mode: {pooling}")
    return selected.detach().float().cpu().numpy()


def _masked_mean(hidden_state, mask, pooling: PoolingMode, batch_index: int = 0):
    if int(mask.sum().item()) == 0:
        raise ValueError(f"{pooling} pooling selected no tokens.")
    return hidden_state[batch_index, mask].mean(dim=0)


def _normalize_poolings(poolings: Sequence[PoolingMode]) -> tuple[PoolingMode, ...]:
    if not poolings:
        raise ValueError("At least one pooling mode must be requested.")
    allowed = {"last_token", "mean_tokens", "text_tokens", "image_tokens"}
    normalized = tuple(str(pooling) for pooling in poolings)
    invalid = [pooling for pooling in normalized if pooling not in allowed]
    if invalid:
        raise ValueError(f"Unsupported pooling modes: {invalid}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Pooling list contains duplicates.")
    return normalized  # type: ignore[return-value]


def layer_tag(layer: int) -> str:
    return f"m{abs(layer)}" if layer < 0 else str(layer)


def _load_dependencies():
    try:
        import torch
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "LLaVA-OneVision extraction requires torch, transformers, and Pillow."
        ) from exc
    return torch, AutoProcessor, LlavaOnevisionForConditionalGeneration
