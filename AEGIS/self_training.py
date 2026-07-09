from __future__ import annotations

import numpy as np


def sample_per_class(
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    """Return a mask containing a reproducible number of trusted rows per class."""
    labels = np.asarray(labels, dtype=np.int64)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if labels.shape != candidate_mask.shape:
        raise ValueError("labels and candidate_mask must have the same shape.")
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive.")

    rng = np.random.default_rng(seed)
    selected_mask = np.zeros(len(labels), dtype=bool)
    for label in (0, 1):
        candidates = np.where(candidate_mask & (labels == label))[0]
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"Class {label} has only {len(candidates)} candidates, "
                f"fewer than requested {samples_per_class}."
            )
        selected = rng.choice(candidates, size=samples_per_class, replace=False)
        selected_mask[selected] = True
    return selected_mask


def select_balanced_pseudo_labels(
    probabilities: np.ndarray,
    candidate_indices: np.ndarray,
    samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select equally sized low- and high-confidence pseudo-label groups."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if probabilities.ndim != 1 or candidate_indices.ndim != 1:
        raise ValueError("probabilities and candidate_indices must be 1D.")
    if len(probabilities) != len(candidate_indices):
        raise ValueError("probabilities and candidate_indices must have the same length.")
    if samples_per_class < 0:
        raise ValueError("samples_per_class cannot be negative.")
    if samples_per_class == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if 2 * samples_per_class > len(candidate_indices):
        raise ValueError("Not enough candidates for disjoint balanced pseudo-label groups.")

    order = np.argsort(probabilities)
    benign_local = order[:samples_per_class]
    malicious_local = order[-samples_per_class:]
    indices = np.concatenate(
        [candidate_indices[benign_local], candidate_indices[malicious_local]]
    )
    labels = np.concatenate(
        [
            np.zeros(samples_per_class, dtype=np.int64),
            np.ones(samples_per_class, dtype=np.int64),
        ]
    )
    return indices, labels


def repeat_trusted_rows(
    trusted_indices: np.ndarray,
    trusted_labels: np.ndarray,
    repeat: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat trusted examples so they remain anchors during pseudo-label training."""
    trusted_indices = np.asarray(trusted_indices, dtype=np.int64)
    trusted_labels = np.asarray(trusted_labels, dtype=np.int64)
    if trusted_indices.shape != trusted_labels.shape:
        raise ValueError("trusted_indices and trusted_labels must have the same shape.")
    if repeat < 1:
        raise ValueError("repeat must be positive.")
    return np.tile(trusted_indices, repeat), np.tile(trusted_labels, repeat)
