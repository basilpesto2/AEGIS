from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from AEGIS.io import align_embeddings_with_metadata, load_embeddings


def require_metadata(metadata: pd.DataFrame | None, name: str = "metadata") -> pd.DataFrame:
    if metadata is None:
        raise ValueError(f"{name} is required.")
    return metadata


def require_column(metadata: pd.DataFrame, column: str, name: str = "metadata") -> None:
    if column not in metadata.columns:
        raise ValueError(f"{name} is missing required column {column!r}.")


def load_aligned_features(
    embedding_path: str | Path,
    metadata: pd.DataFrame,
    id_column: str = "sample_id",
) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings, sample_ids = load_embeddings(embedding_path)
    aligned_embeddings, aligned_metadata, _ = align_embeddings_with_metadata(
        embeddings,
        sample_ids,
        metadata,
        id_column=id_column,
    )
    if aligned_metadata is None:
        raise ValueError("Metadata alignment failed.")
    return aligned_embeddings, aligned_metadata


def load_feature_sets(
    embedding_paths: list[str | Path],
    metadata: pd.DataFrame,
    id_column: str = "sample_id",
) -> list[dict]:
    feature_sets = []
    for embedding_path in sorted(Path(path) for path in embedding_paths):
        embeddings, aligned_metadata = load_aligned_features(
            embedding_path,
            metadata,
            id_column=id_column,
        )
        file_metadata = read_embedding_file_metadata(embedding_path)
        feature_sets.append(
            {
                "name": file_metadata["pooling"] or embedding_path.stem,
                "embedding_file": str(embedding_path),
                "features": embeddings,
                "metadata": aligned_metadata,
                **file_metadata,
            }
        )
    return feature_sets


def unique_by_pooling(feature_sets: list[dict], name: str) -> dict[str, dict]:
    output = {}
    for feature_set in feature_sets:
        pooling = feature_set["pooling"]
        if pooling is None:
            raise ValueError(f"{name} embedding {feature_set['embedding_file']} lacks pooling metadata.")
        if pooling in output:
            raise ValueError(f"{name} embeddings contain duplicate pooling value {pooling!r}.")
        output[str(pooling)] = feature_set
    return output


def read_embedding_file_metadata(path: str | Path) -> dict[str, str | int | None]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "model_id": first_npz_string(data, "model_id"),
            "layer": first_npz_int(data, "layer"),
            "pooling": first_npz_string(data, "pooling"),
        }


def first_npz_string(data, key: str) -> str | None:
    if key not in data:
        return None
    values = np.asarray(data[key]).reshape(-1)
    if len(values) == 0:
        return None
    return str(values[0])


def first_npz_int(data, key: str) -> int | None:
    value = first_npz_string(data, key)
    if value is None:
        return None
    return int(value)
