from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from AEGIS.adapters.llava_onevision import (  # noqa: E402
    LlavaOnevisionExtractionConfig,
    extract_llava_onevision_pooling_embeddings,
)
from AEGIS.adapters.qwen25_vl import (  # noqa: E402
    Qwen25VLExtractionConfig,
    extract_qwen25_vl_pooling_embeddings,
)


MASK_WORDS = ("ignore", "override", "follow", "execute", "provide", "reveal", "output", "comply")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract aligned MLLM representations and perturbation attribution signals.")
    parser.add_argument("--model-family", required=True, choices=("qwen25_vl", "llava_onevision"))
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.metadata)
    if args.model_family == "llava_onevision":
        rows = [row for row in rows if row.get("image_path", "").strip()]
        if not rows:
            raise ValueError("LLaVA extraction requires at least one image-text row")
    adapted = args.output_dir / "aligned_metadata.csv"
    masked = args.output_dir / "aligned_metadata_masked.csv"
    _write_adapter_metadata(adapted, rows, mask=False)
    _write_adapter_metadata(masked, rows, mask=True)

    if args.model_family == "qwen25_vl":
        config = Qwen25VLExtractionConfig(
            model_id=args.model_id or "Qwen/Qwen2.5-VL-3B-Instruct",
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            cache_dir=args.cache_dir,
            layer=args.layer,
            local_files_only=args.local_files_only,
        )
        text_path = extract_qwen25_vl_pooling_embeddings(
            adapted, args.corpus_root, args.output_dir / "raw", ["text_tokens"], config, "original"
        )["text_tokens"]
        masked_path = extract_qwen25_vl_pooling_embeddings(
            masked, args.corpus_root, args.output_dir / "raw", ["text_tokens"], config, "masked"
        )["text_tokens"]
        image_rows = [row for row in rows if row.get("image_path", "").strip()]
        image_embeddings = None
        image_ids: list[str] = []
        image_path = None
        if image_rows:
            image_metadata = args.output_dir / "aligned_metadata_images.csv"
            _write_adapter_metadata(image_metadata, image_rows, mask=False)
            image_path = extract_qwen25_vl_pooling_embeddings(
                image_metadata, args.corpus_root, args.output_dir / "raw", ["image_tokens"], config, "images"
            )["image_tokens"]
            image_embeddings, image_ids = _read_adapter_npz(image_path)
    else:
        config = LlavaOnevisionExtractionConfig(
            model_id=args.model_id or "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            cache_dir=args.cache_dir,
            layer=args.layer,
            local_files_only=args.local_files_only,
        )
        original_paths = extract_llava_onevision_pooling_embeddings(
            adapted, args.corpus_root, args.output_dir / "raw", ["text_tokens", "image_tokens"], config, "original"
        )
        text_path = original_paths["text_tokens"]
        image_path = original_paths["image_tokens"]
        masked_path = extract_llava_onevision_pooling_embeddings(
            masked, args.corpus_root, args.output_dir / "raw", ["text_tokens"], config, "masked"
        )["text_tokens"]
        image_embeddings, image_ids = _read_adapter_npz(image_path)

    text_embeddings, sample_ids = _read_adapter_npz(text_path)
    masked_embeddings, masked_ids = _read_adapter_npz(masked_path)
    if sample_ids != masked_ids:
        raise ValueError("masked extraction IDs do not align with original extraction")
    if image_embeddings is None:
        aligned_images = np.zeros_like(text_embeddings)
    else:
        image_lookup = {sample_id: vector for sample_id, vector in zip(image_ids, image_embeddings)}
        aligned_images = np.vstack([image_lookup.get(sample_id, np.zeros(text_embeddings.shape[1])) for sample_id in sample_ids])
    attribution = _attribution_summary(text_embeddings, masked_embeddings, aligned_images)
    output_path = args.output_dir / "feature_bundle.npz"
    np.savez(
        output_path,
        sample_ids=np.asarray(sample_ids),
        text_embeddings=text_embeddings.astype(np.float32),
        image_embeddings=aligned_images.astype(np.float32),
        attribution_features=attribution.astype(np.float32),
        feature_source=np.asarray([f"{args.model_family}_hidden_states_with_masked_perturbation"]),
    )
    manifest = {
        "model_family": args.model_family,
        "config": asdict(config),
        "rows": len(sample_ids),
        "text_embedding_source": str(text_path),
        "image_embedding_source": str(image_path) if image_path else None,
        "masked_embedding_source": str(masked_path),
        "attribution_method": "aggregate hidden-state delta after fixed instruction-word masking",
        "mask_words": list(MASK_WORDS),
        "feature_bundle": str(output_path),
        "scope_note": "LLaVA output contains image-text rows only because the adapter requires an image.",
    }
    (args.output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "prompt_text", "image_path"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"metadata must contain {sorted(required)}")
    return rows


def _write_adapter_metadata(path: Path, rows: list[dict[str, str]], *, mask: bool) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "text", "image_path"])
        writer.writeheader()
        for row in rows:
            text = row["prompt_text"]
            if mask:
                for word in MASK_WORDS:
                    text = text.replace(word, "[MASK]").replace(word.title(), "[MASK]")
            writer.writerow({"sample_id": row["sample_id"], "text": text, "image_path": row["image_path"]})


def _read_adapter_npz(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float64)
        key = "sample_id" if "sample_id" in data else "sample_ids"
        sample_ids = [str(value) for value in np.asarray(data[key]).reshape(-1)]
    return embeddings, sample_ids


def _attribution_summary(original: np.ndarray, masked: np.ndarray, image: np.ndarray) -> np.ndarray:
    if original.shape != masked.shape or original.shape != image.shape:
        raise ValueError("original, masked, and image arrays must have matching shapes")
    eps = 1e-12
    delta = original - masked
    original_norm = np.linalg.norm(original, axis=1)
    masked_norm = np.linalg.norm(masked, axis=1)
    image_norm = np.linalg.norm(image, axis=1)
    original_masked_cosine = np.sum(original * masked, axis=1) / np.maximum(original_norm * masked_norm, eps)
    text_image_cosine = np.sum(original * image, axis=1) / np.maximum(original_norm * image_norm, eps)
    return np.column_stack(
        [
            np.linalg.norm(delta, axis=1),
            np.mean(np.abs(delta), axis=1),
            np.std(delta, axis=1),
            np.max(np.abs(delta), axis=1),
            original_masked_cosine,
            original_norm,
            masked_norm,
            text_image_cosine,
        ]
    )


if __name__ == "__main__":
    main()
