from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureBundle:
    sample_ids: np.ndarray
    text_embeddings: np.ndarray
    image_embeddings: np.ndarray
    attribution_features: np.ndarray
    feature_source: str


REQUIRED_METADATA = {
    "sample_id",
    "split",
    "group_id",
    "label_id",
    "attack_style",
    "harm_category",
    "modality",
}


def load_metadata(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("metadata is empty")
    missing = REQUIRED_METADATA - set(rows[0])
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id values must be unique")
    if {row["split"] for row in rows} - {"train", "validation", "test"}:
        raise ValueError("split values must be train, validation, or test")
    if {row["label_id"] for row in rows} - {"0", "1"}:
        raise ValueError("label_id values must be 0 or 1")
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
    leaking = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"group leakage across splits: {leaking[:5]}")
    return rows


def load_feature_bundle(path: str | Path, metadata: list[dict[str, str]]) -> FeatureBundle:
    with np.load(path, allow_pickle=False) as data:
        required = {"sample_ids", "text_embeddings", "image_embeddings", "attribution_features", "feature_source"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"feature bundle is missing keys: {sorted(missing)}")
        bundle = FeatureBundle(
            sample_ids=np.asarray(data["sample_ids"]).astype(str),
            text_embeddings=np.asarray(data["text_embeddings"], dtype=np.float64),
            image_embeddings=np.asarray(data["image_embeddings"], dtype=np.float64),
            attribution_features=np.asarray(data["attribution_features"], dtype=np.float64),
            feature_source=str(np.asarray(data["feature_source"]).reshape(-1)[0]),
        )
    expected = np.asarray([row["sample_id"] for row in metadata])
    if not np.array_equal(bundle.sample_ids, expected):
        raise ValueError("feature sample_ids are not exactly aligned to metadata")
    n = len(expected)
    arrays = [bundle.text_embeddings, bundle.image_embeddings, bundle.attribution_features]
    if any(array.ndim != 2 or len(array) != n or not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("feature arrays must be finite 2D arrays aligned to metadata")
    if bundle.text_embeddings.shape != bundle.image_embeddings.shape:
        raise ValueError("text and image embedding shapes must match")
    return bundle
