from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from AEGIS.experiment_io import load_feature_sets, read_embedding_file_metadata
from AEGIS.io import align_embeddings_with_metadata, load_embeddings


def test_load_embeddings_and_align_by_sample_id(tmp_path) -> None:
    embedding_path = tmp_path / "features.npz"
    np.savez(
        embedding_path,
        embeddings=np.asarray([[2.0, 20.0], [1.0, 10.0], [3.0, 30.0]]),
        sample_id=np.asarray(["b", "a", "c"]),
        model_id=np.asarray(["test-model"]),
        layer=np.asarray([-1]),
        pooling=np.asarray(["text_tokens"]),
    )
    metadata = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "label": ["benign", "malicious", "benign"],
        }
    )

    embeddings, sample_ids = load_embeddings(embedding_path)
    aligned, aligned_metadata, aligned_ids = align_embeddings_with_metadata(
        embeddings,
        sample_ids,
        metadata,
    )

    assert aligned_metadata is not None
    assert aligned_ids.tolist() == ["a", "b", "c"]
    assert aligned.tolist() == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]


def test_alignment_rejects_missing_ids(tmp_path) -> None:
    embedding_path = tmp_path / "features.npz"
    np.savez(
        embedding_path,
        embeddings=np.asarray([[1.0], [2.0]]),
        sample_ids=np.asarray(["a", "b"]),
    )
    embeddings, sample_ids = load_embeddings(embedding_path)
    metadata = pd.DataFrame({"sample_id": ["a", "missing"]})

    with pytest.raises(ValueError, match="missing from embeddings"):
        align_embeddings_with_metadata(embeddings, sample_ids, metadata)


def test_experiment_io_reads_provenance_and_feature_sets(tmp_path) -> None:
    embedding_path = tmp_path / "features.npz"
    np.savez(
        embedding_path,
        embeddings=np.asarray([[0.1], [0.9]]),
        sample_id=np.asarray(["s0", "s1"]),
        model_id=np.asarray(["unit-model"]),
        layer=np.asarray([-8]),
        pooling=np.asarray(["image_tokens"]),
    )
    metadata = pd.DataFrame({"sample_id": ["s1", "s0"], "label": ["malicious", "benign"]})

    provenance = read_embedding_file_metadata(embedding_path)
    feature_sets = load_feature_sets([embedding_path], metadata)

    assert provenance == {
        "model_id": "unit-model",
        "layer": -8,
        "pooling": "image_tokens",
    }
    assert feature_sets[0]["name"] == "image_tokens"
    assert feature_sets[0]["features"].tolist() == [[0.9], [0.1]]
