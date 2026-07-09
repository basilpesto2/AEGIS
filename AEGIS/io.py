from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_embeddings(path: str | Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load embeddings and optional sample ids from an NPZ file."""
    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=False) as data:
        if "embeddings" in data:
            embeddings = np.array(data["embeddings"], copy=True)
        else:
            array_keys = [key for key in data.files if data[key].ndim == 2]
            if len(array_keys) != 1:
                raise ValueError(
                    f"{npz_path} must contain an 'embeddings' array or exactly one 2D array."
                )
            embeddings = np.array(data[array_keys[0]], copy=True)

        sample_ids = None
        for key in ("sample_id", "sample_ids", "ids"):
            if key in data:
                sample_ids = np.array(data[key], copy=True).astype(str)
                break

    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}.")
    if sample_ids is not None and len(sample_ids) != embeddings.shape[0]:
        raise ValueError("sample_id length does not match number of embeddings.")

    return embeddings, sample_ids


def load_metadata(path: str | Path | None) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    return pd.read_csv(path)


def align_embeddings_with_metadata(
    embeddings: np.ndarray,
    sample_ids: Optional[np.ndarray],
    metadata: Optional[pd.DataFrame],
    id_column: str = "sample_id",
) -> tuple[np.ndarray, Optional[pd.DataFrame], np.ndarray]:
    """Align embeddings to metadata order when sample ids are available."""
    n_samples = embeddings.shape[0]

    if metadata is None:
        if sample_ids is None:
            sample_ids = np.array([f"sample_{idx:06d}" for idx in range(n_samples)])
        return embeddings, None, sample_ids.astype(str)

    metadata = metadata.copy().reset_index(drop=True)

    if sample_ids is None:
        if len(metadata) != n_samples:
            raise ValueError(
                "Metadata row count must match embeddings when NPZ sample ids are absent."
            )
        if id_column in metadata:
            aligned_ids = metadata[id_column].astype(str).to_numpy()
        else:
            aligned_ids = np.array([f"sample_{idx:06d}" for idx in range(n_samples)])
            metadata[id_column] = aligned_ids
        return embeddings, metadata, aligned_ids

    sample_ids = sample_ids.astype(str)
    if id_column not in metadata:
        if len(metadata) != n_samples:
            raise ValueError(
                f"Metadata is missing '{id_column}' and row count does not match embeddings."
            )
        metadata[id_column] = sample_ids
        return embeddings, metadata, sample_ids

    index_by_id = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    missing = [
        sample_id
        for sample_id in metadata[id_column].astype(str)
        if sample_id not in index_by_id
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Metadata contains sample ids missing from embeddings: {preview}")

    order = [index_by_id[sample_id] for sample_id in metadata[id_column].astype(str)]
    return embeddings[np.asarray(order)], metadata, metadata[id_column].astype(str).to_numpy()


def save_score_table(
    output_path: str | Path,
    sample_ids: np.ndarray,
    scores: np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    predictions: Optional[np.ndarray] = None,
) -> None:
    """Write scores, predictions, and metadata to CSV."""
    if metadata is None:
        table = pd.DataFrame({"sample_id": sample_ids.astype(str)})
    else:
        table = metadata.copy()
        if "sample_id" not in table:
            table.insert(0, "sample_id", sample_ids.astype(str))

    table["svd_maliciousness_score"] = scores
    if predictions is not None:
        table["predicted_malicious"] = predictions.astype(int)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
