from __future__ import annotations

import numpy as np


def cross_modal_consistency_features(
    text_features: np.ndarray,
    image_features: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build compact agreement/disagreement statistics for paired hidden states."""
    text = _matching_features(text_features, image_features)
    image = np.asarray(image_features, dtype=np.float64)
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
            np.log(np.maximum(text_norm, eps) / np.maximum(image_norm, eps)),
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


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return normalized binary entropy in [0, 1]."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be a 1D array.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must be between 0 and 1.")
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return -(
        clipped * np.log2(clipped)
        + (1.0 - clipped) * np.log2(1.0 - clipped)
    )


def _matching_features(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("text and image features must be 2D arrays.")
    if first.shape != second.shape:
        raise ValueError(
            f"text and image feature shapes must match, got {first.shape} and {second.shape}."
        )
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("features contain non-finite values.")
    return first
