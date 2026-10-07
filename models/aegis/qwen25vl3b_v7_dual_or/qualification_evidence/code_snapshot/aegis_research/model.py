from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticDetector:
    """Small deterministic logistic detector with internal standardization."""

    learning_rate: float = 0.05
    epochs: int = 800
    l2: float = 1e-3
    seed: int = 42
    weights_: np.ndarray | None = None
    bias_: float = 0.0
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LogisticDetector":
        x = _as_matrix(features)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if len(x) != len(y) or len(x) == 0:
            raise ValueError("features and labels must contain the same non-zero row count")
        if set(np.unique(y)) - {0.0, 1.0}:
            raise ValueError("labels must be binary values 0 or 1")
        if len(np.unique(y)) != 2:
            raise ValueError("training labels must contain both classes")

        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        z = (x - self.mean_) / self.scale_
        rng = np.random.default_rng(self.seed)
        self.weights_ = rng.normal(0.0, 0.01, size=z.shape[1])
        self.bias_ = 0.0

        positive = max(float(np.sum(y == 1.0)), 1.0)
        negative = max(float(np.sum(y == 0.0)), 1.0)
        sample_weight = np.where(y == 1.0, len(y) / (2.0 * positive), len(y) / (2.0 * negative))
        for _ in range(self.epochs):
            probabilities = _sigmoid(z @ self.weights_ + self.bias_)
            error = (probabilities - y) * sample_weight
            gradient = z.T @ error / len(z) + self.l2 * self.weights_
            bias_gradient = float(error.mean())
            self.weights_ -= self.learning_rate * gradient
            self.bias_ -= self.learning_rate * bias_gradient
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("detector is not fitted")
        x = _as_matrix(features)
        if x.shape[1] != len(self.weights_):
            raise ValueError(f"expected {len(self.weights_)} features, received {x.shape[1]}")
        return _sigmoid(((x - self.mean_) / self.scale_) @ self.weights_ + self.bias_)

    def save(self, path: str, *, threshold: float, metadata: dict[str, object]) -> None:
        if self.weights_ is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("detector is not fitted")
        import json

        np.savez(
            path,
            weights=self.weights_,
            bias=np.asarray([self.bias_]),
            mean=self.mean_,
            scale=self.scale_,
            threshold=np.asarray([threshold]),
            learning_rate=np.asarray([self.learning_rate]),
            epochs=np.asarray([self.epochs]),
            l2=np.asarray([self.l2]),
            seed=np.asarray([self.seed]),
            metadata_json=np.asarray([json.dumps(metadata, sort_keys=True)]),
        )


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    """Select a validation threshold by F1, then precision, then higher threshold."""

    from .metrics import classification_metrics

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
    ranked: list[tuple[tuple[float, float, float], float, dict[str, float]]] = []
    for threshold in candidates:
        metrics = classification_metrics(y, p, float(threshold))
        ranked.append(((metrics["f1"], metrics["precision"], float(threshold)), float(threshold), metrics))
    _, threshold, metrics = max(ranked, key=lambda item: item[0])
    threshold = min(max(threshold, 1e-6), 1.0 - 1e-6)
    return threshold, metrics


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _as_matrix(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("features must be a finite two-dimensional array")
    return x
