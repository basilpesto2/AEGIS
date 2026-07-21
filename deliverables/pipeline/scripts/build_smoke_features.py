from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES_ROOT = PIPELINE_ROOT.parent
DEFAULT_METADATA = DELIVERABLES_ROOT / "annotated_benchmark" / "data" / "benchmark.csv"
DEFAULT_OUTPUT = PIPELINE_ROOT / "fixtures" / "smoke_features.npz"
DIMENSION = 32


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic non-MLLM smoke-test features.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()
    with args.metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("benchmark metadata is empty")
    id_column = "sample_id" if "sample_id" in rows[0] else "variant_id"
    if id_column not in rows[0] or "prompt_text" not in rows[0] or "image_path" not in rows[0]:
        raise ValueError("metadata must contain sample_id or variant_id, prompt_text, and image_path")
    asset_root = args.asset_root or (args.metadata.parent.parent if args.metadata.parent.name == "data" else args.metadata.parent)
    text_embeddings = np.vstack([_hashed_embedding(row["prompt_text"]) for row in rows])
    image_embeddings = np.vstack([_image_embedding(row, asset_root) for row in rows])
    attribution = np.vstack(
        [_attribution_features(row["prompt_text"], text_embeddings[i], image_embeddings[i]) for i, row in enumerate(rows)]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        sample_ids=np.asarray([row[id_column] for row in rows]),
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        attribution_features=attribution,
        feature_source=np.asarray(["deterministic_smoke_fixture_not_mllm_evidence"]),
    )
    manifest = {
        "feature_source": "deterministic_smoke_fixture_not_mllm_evidence",
        "purpose": "end-to-end software verification only",
        "rows": len(rows),
        "id_column": id_column,
        "asset_root": str(asset_root),
        "text_dimension": DIMENSION,
        "image_dimension": DIMENSION,
        "attribution_dimension": int(attribution.shape[1]),
        "metadata_sha256": _sha256(args.metadata),
        "bundle_sha256": _sha256(args.output),
        "extractor": str(Path(__file__).relative_to(DELIVERABLES_ROOT.parent)),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _hashed_embedding(text: str) -> np.ndarray:
    normalized = " ".join(text.lower().split())
    padded = f"  {normalized}  "
    vector = np.zeros(DIMENSION, dtype=np.float64)
    for width in (2, 3, 4):
        for index in range(len(padded) - width + 1):
            token = padded[index : index + width].encode("utf-8")
            digest = hashlib.blake2b(token, digest_size=8, person=b"aegis-v1").digest()
            bucket = int.from_bytes(digest[:4], "little") % DIMENSION
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _image_embedding(row: dict[str, str], benchmark_root: Path) -> np.ndarray:
    if not row["image_path"]:
        return np.zeros(DIMENSION, dtype=np.float64)
    path = benchmark_root / row["image_path"]
    with Image.open(path) as image:
        values = np.asarray(image.convert("L").resize((8, 4)), dtype=np.float64).reshape(-1)
    values = (values - values.mean()) / max(values.std(), 1.0)
    norm = np.linalg.norm(values)
    return values / norm if norm else values


def _attribution_features(text: str, text_embedding: np.ndarray, image_embedding: np.ndarray) -> np.ndarray:
    lowered = text.lower()
    instruction_words = ("ignore", "follow", "execute", "provide", "reveal", "output", "override", "comply")
    masked = lowered
    for word in instruction_words:
        masked = masked.replace(word, "")
    perturbation = _hashed_embedding(masked)
    text_norm = float(np.linalg.norm(text_embedding))
    image_norm = float(np.linalg.norm(image_embedding))
    cosine = float(np.dot(text_embedding, image_embedding) / max(text_norm * image_norm, 1e-12))
    return np.asarray(
        [
            min(len(text) / 240.0, 2.0),
            lowered.count("[redacted") / 2.0,
            sum(lowered.count(word) for word in instruction_words) / 4.0,
            sum(character.isupper() for character in text) / max(len(text), 1),
            text.count("[") / 4.0,
            float(np.linalg.norm(text_embedding - perturbation)),
            image_norm,
            cosine,
        ],
        dtype=np.float64,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.path.insert(0, str(PIPELINE_ROOT))
    main()
