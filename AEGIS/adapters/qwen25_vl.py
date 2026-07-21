from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image

from AEGIS.provenance import preprocessing_fingerprint


PoolingMode = Literal["last_token", "mean_tokens", "text_tokens", "image_tokens"]


@dataclass(frozen=True)
class Qwen25VLExtractionConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    cache_dir: str | None = "models/huggingface"
    layer: int = -1
    pooling: PoolingMode = "last_token"
    add_generation_prompt: bool = False
    torch_dtype: str = "auto"
    device_map: str = "auto"
    min_pixels: int | None = None
    max_pixels: int | None = None
    max_samples: int | None = None
    local_files_only: bool = False


def extract_qwen25_vl_layer_embeddings(
    metadata_path: str | Path,
    corpus_root: str | Path,
    output_dir: str | Path,
    layers: Sequence[int],
    config: Qwen25VLExtractionConfig | None = None,
    output_prefix: str | None = None,
) -> dict[int, Path]:
    """Extract several Qwen2.5-VL layers in one model pass.

    Each selected layer is saved as a normal AEGIS embeddings NPZ, so downstream
    scoring scripts can consume the files without special multi-layer handling.
    """
    layer_ids = _normalize_layers(layers)
    config = config or Qwen25VLExtractionConfig()
    arrays, sample_ids = _extract_qwen25_vl_arrays(
        metadata_path=metadata_path,
        corpus_root=corpus_root,
        config=config,
        layers=layer_ids,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or Path(metadata_path).stem

    paths: dict[int, Path] = {}
    for layer in layer_ids:
        output_path = output_root / f"{prefix}_layer_{layer_tag(layer)}_{config.pooling}.npz"
        _save_embedding_npz(
            output_path=output_path,
            embeddings=np.vstack(arrays[layer]).astype(np.float32),
            sample_ids=sample_ids,
            model_id=config.model_id,
            model_revision=config.model_revision,
            tokenizer_revision=config.tokenizer_revision,
            preprocessing_sha256=qwen_preprocessing_sha256(config),
            layer=layer,
            pooling=config.pooling,
        )
        paths[layer] = output_path
    return paths


def extract_qwen25_vl_pooling_embeddings(
    metadata_path: str | Path,
    corpus_root: str | Path,
    output_dir: str | Path,
    poolings: Sequence[PoolingMode],
    config: Qwen25VLExtractionConfig | None = None,
    output_prefix: str | None = None,
) -> dict[PoolingMode, Path]:
    """Extract several pooling views from one layer in a single model pass."""
    pooling_ids = _normalize_poolings(poolings)
    config = config or Qwen25VLExtractionConfig()
    arrays, sample_ids = _extract_qwen25_vl_feature_arrays(
        metadata_path=metadata_path,
        corpus_root=corpus_root,
        config=config,
        layers=(config.layer,),
        poolings=pooling_ids,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or Path(metadata_path).stem
    paths: dict[PoolingMode, Path] = {}
    for pooling in pooling_ids:
        output_path = output_root / f"{prefix}_layer_{layer_tag(config.layer)}_{pooling}.npz"
        _save_embedding_npz(
            output_path=output_path,
            embeddings=np.vstack(arrays[(config.layer, pooling)]).astype(np.float32),
            sample_ids=sample_ids,
            model_id=config.model_id,
            model_revision=config.model_revision,
            tokenizer_revision=config.tokenizer_revision,
            preprocessing_sha256=qwen_preprocessing_sha256(config),
            layer=config.layer,
            pooling=pooling,
        )
        paths[pooling] = output_path
    return paths


def extract_qwen25_vl_embeddings(
    metadata_path: str | Path,
    corpus_root: str | Path,
    output_path: str | Path,
    config: Qwen25VLExtractionConfig | None = None,
) -> None:
    """Extract prompt embeddings from Qwen2.5-VL into AEGIS NPZ format.

    This adapter intentionally does not generate model responses. It forwards the
    prompt through the MLLM and pools hidden states from the requested layer.
    """
    config = config or Qwen25VLExtractionConfig()
    arrays, sample_ids = _extract_qwen25_vl_arrays(
        metadata_path=metadata_path,
        corpus_root=corpus_root,
        config=config,
        layers=(config.layer,),
    )

    _save_embedding_npz(
        output_path=output_path,
        embeddings=np.vstack(arrays[config.layer]).astype(np.float32),
        sample_ids=sample_ids,
        model_id=config.model_id,
        model_revision=config.model_revision,
        tokenizer_revision=config.tokenizer_revision,
        preprocessing_sha256=qwen_preprocessing_sha256(config),
        layer=config.layer,
        pooling=config.pooling,
    )


def _extract_qwen25_vl_arrays(
    metadata_path: str | Path,
    corpus_root: str | Path,
    config: Qwen25VLExtractionConfig,
    layers: Sequence[int],
) -> tuple[dict[int, list[np.ndarray]], list[str]]:
    feature_arrays, sample_ids = _extract_qwen25_vl_feature_arrays(
        metadata_path=metadata_path,
        corpus_root=corpus_root,
        config=config,
        layers=layers,
        poolings=(config.pooling,),
    )
    return (
        {layer: feature_arrays[(layer, config.pooling)] for layer in _normalize_layers(layers)},
        sample_ids,
    )


def _extract_qwen25_vl_feature_arrays(
    metadata_path: str | Path,
    corpus_root: str | Path,
    config: Qwen25VLExtractionConfig,
    layers: Sequence[int],
    poolings: Sequence[PoolingMode],
) -> tuple[dict[tuple[int, PoolingMode], list[np.ndarray]], list[str]]:
    layer_ids = _normalize_layers(layers)
    pooling_ids = _normalize_poolings(poolings)

    metadata = pd.read_csv(metadata_path)
    if "sample_id" not in metadata or "text" not in metadata:
        raise ValueError("Metadata must contain 'sample_id' and 'text' columns.")
    if "image_path" not in metadata:
        metadata["image_path"] = ""

    if config.max_samples is not None:
        metadata = metadata.head(config.max_samples).copy()

    torch, processor, model, special_token_ids, process_vision_info = _load_qwen_runtime(
        config.model_id,
        config.cache_dir,
        config.torch_dtype,
        config.device_map,
        config.min_pixels,
        config.max_pixels,
        config.model_revision,
        config.tokenizer_revision,
        config.local_files_only,
    )

    root = Path(corpus_root)
    embeddings_by_feature: dict[tuple[int, PoolingMode], list[np.ndarray]] = {
        (layer, pooling): [] for layer in layer_ids for pooling in pooling_ids
    }
    sample_ids: list[str] = []

    with torch.no_grad():
        for row in metadata.itertuples(index=False):
            messages = [_build_message(row, root)]
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=config.add_generation_prompt,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states.")
            for layer in layer_ids:
                layer_hidden = hidden_states[layer]
                for pooling in pooling_ids:
                    embedding = _pool_hidden_state(
                        layer_hidden,
                        inputs.get("attention_mask"),
                        input_ids=inputs.get("input_ids"),
                        special_token_ids=special_token_ids,
                        pooling=pooling,
                    )
                    embeddings_by_feature[(layer, pooling)].append(embedding)
            sample_ids.append(str(getattr(row, "sample_id")))

    return embeddings_by_feature, sample_ids


@lru_cache(maxsize=4)
def _load_qwen_runtime(
    model_id: str,
    cache_dir: str | None,
    torch_dtype: str,
    device_map: str,
    min_pixels: int | None,
    max_pixels: int | None,
    model_revision: str | None,
    tokenizer_revision: str | None,
    local_files_only: bool = False,
):
    """Load and retain one Qwen runtime per model/preprocessing configuration."""
    torch, AutoProcessor, QwenModel, process_vision_info = _load_qwen_dependencies()
    processor_kwargs = {}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        revision=tokenizer_revision or model_revision,
        local_files_only=local_files_only,
        **processor_kwargs,
    )
    model = QwenModel.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        cache_dir=cache_dir,
        revision=model_revision,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return torch, processor, model, _qwen_special_token_ids(processor), process_vision_info


def clear_qwen_runtime_cache() -> None:
    """Release cached Qwen runtime references during shutdown or tests."""
    _load_qwen_runtime.cache_clear()


def qwen_runtime_cache_info() -> dict[str, int]:
    return dict(_load_qwen_runtime.cache_info()._asdict())


def _save_embedding_npz(
    output_path: str | Path,
    embeddings: np.ndarray,
    sample_ids: Sequence[str],
    model_id: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    preprocessing_sha256: str,
    layer: int,
    pooling: PoolingMode,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        embeddings=embeddings,
        sample_id=np.asarray(sample_ids),
        model_id=np.asarray([model_id]),
        model_family=np.asarray(["qwen25_vl"]),
        model_revision=np.asarray([model_revision or ""]),
        tokenizer_revision=np.asarray([tokenizer_revision or model_revision or ""]),
        preprocessing_sha256=np.asarray([preprocessing_sha256]),
        layer=np.asarray([layer]),
        pooling=np.asarray([pooling]),
    )


def qwen_preprocessing_sha256(config: Qwen25VLExtractionConfig) -> str:
    return preprocessing_fingerprint(
        "qwen25_vl",
        adapter_schema=1,
        add_generation_prompt=config.add_generation_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        padding=True,
        chat_template=True,
    )


def _build_message(row, root: Path) -> dict:
    content = []
    image_path = getattr(row, "image_path", "")
    if not pd.isna(image_path) and str(image_path).strip():
        full_path = root / str(image_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {full_path}")
        # Open once to fail early on corrupt images, then pass a local path to Qwen utils.
        with Image.open(full_path) as image:
            image.verify()
        content.append({"type": "image", "image": str(full_path)})

    content.append({"type": "text", "text": str(getattr(row, "text"))})
    return {"role": "user", "content": content}


def _pool_hidden_state(
    hidden_state,
    attention_mask,
    pooling: PoolingMode,
    input_ids=None,
    special_token_ids: dict[str, int] | None = None,
) -> np.ndarray:
    if pooling == "last_token":
        if attention_mask is None:
            token_index = hidden_state.shape[1] - 1
        else:
            token_index = int(attention_mask[0].sum().item()) - 1
        pooled = hidden_state[0, token_index]
    elif pooling == "mean_tokens":
        mask = _attention_mask(hidden_state, attention_mask)
        pooled = _masked_mean(hidden_state, mask, pooling)
    elif pooling == "text_tokens":
        mask = _text_token_mask(hidden_state, attention_mask, input_ids, special_token_ids)
        pooled = _masked_mean(hidden_state, mask, pooling)
    elif pooling == "image_tokens":
        mask = _image_token_mask(hidden_state, attention_mask, input_ids, special_token_ids)
        pooled = _masked_mean(hidden_state, mask, pooling)
    else:
        raise ValueError(f"Unsupported pooling mode: {pooling}")

    return pooled.detach().float().cpu().numpy()


def _attention_mask(hidden_state, attention_mask):
    if attention_mask is None:
        return hidden_state.new_ones(hidden_state.shape[1], dtype=bool)
    return attention_mask[0].to(hidden_state.device).bool()


def _text_token_mask(hidden_state, attention_mask, input_ids, special_token_ids):
    if input_ids is None:
        raise ValueError("text_tokens pooling requires input_ids.")
    if special_token_ids is None:
        raise ValueError("text_tokens pooling requires Qwen special token ids.")

    mask = _attention_mask(hidden_state, attention_mask)
    ids = input_ids[0].to(hidden_state.device)
    excluded = [
        "image_pad",
        "video_pad",
        "vision_start",
        "vision_end",
    ]
    for name in excluded:
        token_id = special_token_ids.get(name)
        if token_id is not None:
            mask = mask & (ids != token_id)
    return mask


def _image_token_mask(hidden_state, attention_mask, input_ids, special_token_ids):
    if input_ids is None:
        raise ValueError("image_tokens pooling requires input_ids.")
    if special_token_ids is None or special_token_ids.get("image_pad") is None:
        raise ValueError("image_tokens pooling requires a Qwen image token id.")

    mask = _attention_mask(hidden_state, attention_mask)
    ids = input_ids[0].to(hidden_state.device)
    return mask & (ids == special_token_ids["image_pad"])


def _masked_mean(hidden_state, mask, pooling: PoolingMode):
    if int(mask.sum().item()) == 0:
        raise ValueError(f"{pooling} pooling selected no tokens.")
    return hidden_state[0, mask].mean(dim=0)


def _normalize_layers(layers: Sequence[int]) -> tuple[int, ...]:
    if not layers:
        raise ValueError("At least one layer must be requested.")
    normalized = tuple(int(layer) for layer in layers)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Layer list contains duplicates.")
    return normalized


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


def _qwen_special_token_ids(processor) -> dict[str, int | None]:
    tokenizer = processor.tokenizer
    tokens = {
        "image_pad": "<|image_pad|>",
        "video_pad": "<|video_pad|>",
        "vision_start": "<|vision_start|>",
        "vision_end": "<|vision_end|>",
    }
    ids: dict[str, int | None] = {}
    for name, token in tokens.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        ids[name] = None if token_id == tokenizer.unk_token_id else int(token_id)
    return ids


def _load_qwen_dependencies():
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Qwen2.5-VL extraction requires optional ML dependencies. "
            "Install them with `python -m pip install 'aegis-mllm-guard[mllm]'`."
        ) from exc

    return torch, AutoProcessor, Qwen2_5_VLForConditionalGeneration, process_vision_info
