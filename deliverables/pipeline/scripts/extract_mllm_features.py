from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
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
    parser.add_argument(
        "--runtime-model-id",
        help=(
            "Optional local LLaVA checkpoint path used for loading while --model-id "
            "retains the detector's provenance identifier."
        ),
    )
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace files in a non-empty output directory.")
    args = parser.parse_args()
    if not args.model_revision or not args.tokenizer_revision:
        parser.error(
            "--model-revision and --tokenizer-revision are required for "
            "provenance-complete MLLM extraction"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {args.output_dir}; "
            "choose a versioned --output-dir or use --force"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.metadata)
    id_column = _id_column(rows[0])
    if args.model_family == "llava_onevision":
        rows = [row for row in rows if row.get("image_path", "").strip()]
        if not rows:
            raise ValueError("LLaVA extraction requires at least one image-text row")
    aligned_source = args.output_dir / "aligned_source_metadata.csv"
    adapted = args.output_dir / "aligned_metadata.csv"
    masked = args.output_dir / "aligned_metadata_masked.csv"
    _write_source_metadata(aligned_source, rows)
    _write_adapter_metadata(adapted, rows, id_column=id_column, mask=False)
    _write_adapter_metadata(masked, rows, id_column=id_column, mask=True)

    if args.model_family == "qwen25_vl":
        if args.runtime_model_id:
            parser.error("--runtime-model-id is supported only for llava_onevision")
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
            _write_adapter_metadata(
                image_metadata,
                image_rows,
                id_column=id_column,
                mask=False,
            )
            image_path = extract_qwen25_vl_pooling_embeddings(
                image_metadata, args.corpus_root, args.output_dir / "raw", ["image_tokens"], config, "images"
            )["image_tokens"]
            image_embeddings, image_ids = _read_adapter_npz(image_path)
    else:
        config = LlavaOnevisionExtractionConfig(
            model_id=args.model_id or "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            runtime_model_id=args.runtime_model_id,
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
    embedding_provenance = _read_embedding_provenance(text_path)
    _require_matching_embedding_provenance(
        embedding_provenance,
        _read_embedding_provenance(masked_path),
        "masked extraction",
    )
    if image_path is not None:
        _require_matching_embedding_provenance(
            embedding_provenance,
            _read_embedding_provenance(image_path),
            "image extraction",
            allow_pooling_difference=True,
        )
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
        model_family=np.asarray([embedding_provenance["model_family"]]),
        model_id=np.asarray([embedding_provenance["model_id"]]),
        model_revision=np.asarray([embedding_provenance["model_revision"]]),
        tokenizer_revision=np.asarray(
            [embedding_provenance["tokenizer_revision"]]
        ),
        preprocessing_sha256=np.asarray(
            [embedding_provenance["preprocessing_sha256"]]
        ),
        layer=np.asarray([embedding_provenance["layer"]], dtype=np.int64),
        pooling=np.asarray([embedding_provenance["pooling"]]),
        id_column=np.asarray([id_column]),
    )
    bundle_preparation = {
        "adapter": args.model_family,
        "layer": args.layer,
        "pooling": ["text_tokens", "image_tokens"],
        "attribution": "fixed_instruction_word_masking",
        "mask_words": list(MASK_WORDS),
        "corpus_metadata_sha256": _sha256(args.metadata),
    }
    manifest = {
        "schema_version": 2,
        "model_family": args.model_family,
        "model_id": embedding_provenance["model_id"],
        "model_revision": embedding_provenance["model_revision"],
        "tokenizer_revision": embedding_provenance["tokenizer_revision"],
        "preprocessing_sha256": embedding_provenance["preprocessing_sha256"],
        "layer": embedding_provenance["layer"],
        "pooling": embedding_provenance["pooling"],
        "id_column": id_column,
        "config": asdict(config),
        "rows": len(sample_ids),
        "aligned_source_metadata": _portable_path(aligned_source),
        "aligned_source_metadata_sha256": _sha256(aligned_source),
        "text_embedding_source": _portable_path(text_path),
        "image_embedding_source": _portable_path(image_path) if image_path else None,
        "masked_embedding_source": _portable_path(masked_path),
        "attribution_method": "aggregate hidden-state delta after fixed instruction-word masking",
        "mask_words": list(MASK_WORDS),
        "available_pooling_views": ["text_tokens", "image_tokens"],
        "embedding_dimension": int(text_embeddings.shape[1]),
        "bundle_preparation": bundle_preparation,
        "bundle_preparation_sha256": hashlib.sha256(
            json.dumps(
                bundle_preparation,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "feature_bundle": _portable_path(output_path),
        "feature_bundle_sha256": _sha256(output_path),
        "metadata": _portable_path(args.metadata),
        "metadata_sha256": _sha256(args.metadata),
        "toolchain": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "scope_note": "LLaVA output contains image-text rows only because the adapter requires an image.",
    }
    (args.output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"prompt_text", "image_path"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"metadata must contain {sorted(required)}")
    id_column = _id_column(rows[0])
    identifiers = [row[id_column].strip() for row in rows]
    if any(not value for value in identifiers):
        raise ValueError(f"{id_column} values must be non-empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{id_column} values must be unique")
    return rows


def _id_column(row: dict[str, str]) -> str:
    if "sample_id" in row:
        return "sample_id"
    if "variant_id" in row:
        return "variant_id"
    raise ValueError("metadata must contain sample_id or variant_id")


def _write_source_metadata(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    if not rows:
        raise ValueError("cannot write empty aligned metadata")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_adapter_metadata(
    path: Path,
    rows: list[dict[str, str]],
    *,
    id_column: str,
    mask: bool,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "text", "image_path"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            text = row["prompt_text"]
            if mask:
                for word in MASK_WORDS:
                    text = text.replace(word, "[MASK]").replace(word.title(), "[MASK]")
            writer.writerow(
                {
                    "sample_id": row[id_column],
                    "text": text,
                    "image_path": row["image_path"],
                }
            )


def _read_adapter_npz(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float64)
        key = "sample_id" if "sample_id" in data else "sample_ids"
        sample_ids = [str(value) for value in np.asarray(data[key]).reshape(-1)]
    return embeddings, sample_ids


def _read_embedding_provenance(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "model_family",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "preprocessing_sha256",
            "layer",
            "pooling",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"embedding output lacks provenance ({path}): "
                + ", ".join(sorted(missing))
            )
        result: dict[str, object] = {
            key: (
                int(np.asarray(data[key]).reshape(-1)[0])
                if key == "layer"
                else str(np.asarray(data[key]).reshape(-1)[0]).strip()
            )
            for key in required
        }
    for key in required - {"layer"}:
        if not result[key]:
            raise ValueError(f"embedding provenance {key} must be non-empty")
    fingerprint = str(result["preprocessing_sha256"])
    if (
        len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("embedding preprocessing_sha256 must be lowercase SHA-256 hex")
    return result


def _require_matching_embedding_provenance(
    expected: dict[str, object],
    actual: dict[str, object],
    label: str,
    *,
    allow_pooling_difference: bool = False,
) -> None:
    ignored = {"pooling"} if allow_pooling_difference else set()
    mismatches = [
        key
        for key in expected
        if key not in ignored and expected[key] != actual[key]
    ]
    if mismatches:
        raise ValueError(
            f"{label} provenance does not match text extraction: "
            + ", ".join(sorted(mismatches))
        )


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    main()
