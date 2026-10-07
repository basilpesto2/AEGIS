from __future__ import annotations

import numpy as np


POOLING_FEATURE_VIEWS = {
    "text_tokens": "text_representation",
    "image_tokens": "image_representation",
    "text_image_tokens": "joint_representation",
}


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    return -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))


def cross_modal_consistency(text: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    text = _matching(text, image)
    image = np.asarray(image, dtype=np.float64)
    eps = 1e-12
    text_norm = np.linalg.norm(text, axis=1)
    image_norm = np.linalg.norm(image, axis=1)
    text_unit = text / np.maximum(text_norm[:, None], eps)
    image_unit = image / np.maximum(image_norm[:, None], eps)
    difference = text_unit - image_unit
    product = text_unit * image_unit
    features = np.column_stack(
        [
            np.sum(text_unit * image_unit, axis=1),
            np.linalg.norm(difference, axis=1),
            np.mean(np.abs(difference), axis=1),
            np.std(difference, axis=1),
            np.max(np.abs(difference), axis=1),
            np.mean(product, axis=1),
            np.std(product, axis=1),
            np.log((text_norm + eps) / (image_norm + eps)),
        ]
    )
    names = [
        "cosine_similarity",
        "unit_l2_distance",
        "mean_absolute_difference",
        "difference_std",
        "max_absolute_difference",
        "mean_elementwise_product",
        "product_std",
        "log_norm_ratio",
    ]
    return features, names


def build_feature_views(
    text: np.ndarray,
    image: np.ndarray,
    attribution: np.ndarray,
) -> dict[str, np.ndarray]:
    text = _matching(text, image)
    attribution = np.asarray(attribution, dtype=np.float64)
    if attribution.ndim != 2 or len(attribution) != len(text):
        raise ValueError("attribution features must be a 2D array aligned to embeddings")
    consistency, _ = cross_modal_consistency(text, image)
    return {
        "text_representation": text,
        "image_representation": image,
        "joint_representation": np.concatenate([text, image], axis=1),
        "cross_modal_consistency": consistency,
        "attribution": attribution,
        "joint_plus_consistency": np.concatenate([text, image, consistency], axis=1),
        "all_input_signals": np.concatenate([text, image, consistency, attribution], axis=1),
    }


def _matching(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ValueError("text and image embeddings must be finite 2D arrays with matching shapes")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("embeddings contain non-finite values")
    return first
