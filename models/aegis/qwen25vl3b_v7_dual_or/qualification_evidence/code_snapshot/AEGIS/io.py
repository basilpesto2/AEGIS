from __future__ import annotations

from pathlib import Path

import numpy as np


def load_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load embeddings and optional sample IDs from an NPZ file."""

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
