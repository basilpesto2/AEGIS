from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SVDMaliciousnessDetector:
    """VLMGuard-style weighted projection detector.

    The detector is fit on an unlabeled mixture. Higher scores indicate stronger
    alignment with the top singular-vector subspace discovered from the mixture.
    """

    n_components: int = 8
    center: bool = True
    weight_by_singular_value: bool = True

    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    singular_values_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> "SVDMaliciousnessDetector":
        x = _as_2d_float(embeddings)
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1.")

        self.mean_ = x.mean(axis=0) if self.center else np.zeros(x.shape[1])
        centered = x - self.mean_

        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        keep = min(self.n_components, vt.shape[0])
        self.components_ = vt[:keep]
        self.singular_values_ = singular_values[:keep]
        return self

    def score_samples(self, embeddings: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None or self.singular_values_ is None:
            raise RuntimeError("Detector must be fit before scoring.")

        x = _as_2d_float(embeddings)
        if x.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"Expected embedding dimension {self.mean_.shape[0]}, got {x.shape[1]}."
            )

        centered = x - self.mean_
        projections = centered @ self.components_.T
        squared_projection = projections**2
        if self.weight_by_singular_value:
            squared_projection = squared_projection * self.singular_values_
        return np.mean(squared_projection, axis=1)

    def threshold_from_prior(self, scores: np.ndarray, malicious_prior: float) -> float:
        """Choose threshold so approximately `malicious_prior` score highest samples are positive."""
        if not 0.0 < malicious_prior < 1.0:
            raise ValueError("malicious_prior must be between 0 and 1.")
        return float(np.quantile(np.asarray(scores), 1.0 - malicious_prior))

    def predict_from_prior(self, scores: np.ndarray, malicious_prior: float) -> np.ndarray:
        threshold = self.threshold_from_prior(scores, malicious_prior)
        return (np.asarray(scores) >= threshold).astype(np.int64)


def pseudo_labels_from_scores(
    scores: np.ndarray,
    malicious_prior: float,
    benign_keep_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy pseudo-labels from SVD scores.

    Returns `(indices, labels)`. High-score samples become candidate malicious.
    Low-score samples become candidate benign. If `benign_keep_fraction` is
    omitted, all samples below the malicious threshold are kept as benign.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array.")
    if not 0.0 < malicious_prior < 1.0:
        raise ValueError("malicious_prior must be between 0 and 1.")

    high_threshold = float(np.quantile(scores, 1.0 - malicious_prior))
    malicious = np.where(scores >= high_threshold)[0]

    if benign_keep_fraction is None:
        benign = np.where(scores < high_threshold)[0]
    else:
        if not 0.0 < benign_keep_fraction <= 1.0:
            raise ValueError("benign_keep_fraction must be in (0, 1].")
        low_threshold = float(np.quantile(scores, benign_keep_fraction))
        benign = np.where(scores <= low_threshold)[0]

    indices = np.concatenate([benign, malicious])
    labels = np.concatenate(
        [
            np.zeros(len(benign), dtype=np.int64),
            np.ones(len(malicious), dtype=np.int64),
        ]
    )
    return indices, labels


def _as_2d_float(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Embeddings contain non-finite values.")
    return x

