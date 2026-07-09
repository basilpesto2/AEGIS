from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an AEGIS NPZ embedding panel by sample_id from multiple existing NPZ files."
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--embedding-files", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="sample_id")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    if args.id_column not in metadata.columns:
        raise ValueError(f"Metadata is missing id column {args.id_column!r}.")

    embedding_index: dict[str, np.ndarray] = {}
    file_metadata = []
    feature_dim = None
    for embedding_file in args.embedding_files:
        path = Path(embedding_file)
        with np.load(path, allow_pickle=False) as data:
            if "embeddings" not in data:
                raise ValueError(f"{path} is missing an 'embeddings' array.")
            ids = _load_ids(data, path)
            embeddings = np.asarray(data["embeddings"], dtype=np.float64)
            if feature_dim is None:
                feature_dim = embeddings.shape[1]
            elif embeddings.shape[1] != feature_dim:
                raise ValueError(
                    f"{path} has feature dim {embeddings.shape[1]}, expected {feature_dim}."
                )
            for sample_id, embedding in zip(ids, embeddings, strict=True):
                if sample_id not in embedding_index:
                    embedding_index[sample_id] = embedding
            file_metadata.append(_read_metadata(data))

    sample_ids = np.asarray(metadata[args.id_column].astype(str).to_numpy(), dtype=str)
    missing = [sample_id for sample_id in sample_ids if sample_id not in embedding_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} metadata rows are missing embeddings: {preview}")

    output_embeddings = np.stack([embedding_index[sample_id] for sample_id in sample_ids], axis=0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_values = _merge_metadata(file_metadata)
    np.savez_compressed(
        output_path,
        embeddings=output_embeddings,
        sample_id=sample_ids,
        model_id=np.asarray([metadata_values["model_id"]], dtype=str),
        layer=np.array([metadata_values["layer"]], dtype=np.int64),
        pooling=np.asarray([metadata_values["pooling"]], dtype=str),
    )

    print(f"Wrote {output_embeddings.shape} embeddings to {args.output}")
    print(f"pooling={metadata_values['pooling']}, layer={metadata_values['layer']}")


def _load_ids(data: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
    for key in ("sample_id", "sample_ids", "ids"):
        if key in data:
            return data[key].astype(str)
    raise ValueError(f"{path} does not contain sample IDs.")


def _read_metadata(data: np.lib.npyio.NpzFile) -> dict[str, str | int]:
    return {
        "model_id": _first_string(data, "model_id", default="mixed"),
        "layer": _first_int(data, "layer", default=-1),
        "pooling": _first_string(data, "pooling", default="unknown"),
    }


def _merge_metadata(items: list[dict[str, str | int]]) -> dict[str, str | int]:
    merged = {}
    for key in ("model_id", "layer", "pooling"):
        values = {item[key] for item in items}
        if len(values) == 1:
            merged[key] = values.pop()
        elif key == "model_id":
            merged[key] = "mixed"
        else:
            raise ValueError(f"Input embeddings have inconsistent {key}: {sorted(values)}")
    return merged


def _first_string(data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in data or len(data[key]) == 0:
        return default
    return str(data[key][0])


def _first_int(data: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    if key not in data or len(data[key]) == 0:
        return default
    return int(data[key][0])


if __name__ == "__main__":
    main()
