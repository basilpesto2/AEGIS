from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticRegressionNumpy:
    """Small dependency-free logistic regression for pseudo-labeled embeddings."""

    learning_rate: float = 0.1
    epochs: int = 500
    l2: float = 1e-4
    standardize: bool = True
    random_seed: int = 42

    weights_: np.ndarray | None = None
    bias_: float = 0.0
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LogisticRegressionNumpy":
        x = _as_2d_float(features)
        y = np.asarray(labels, dtype=np.float64)
        if y.ndim != 1 or len(y) != x.shape[0]:
            raise ValueError("labels must be 1D and match feature rows.")
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("labels must contain only 0 and 1.")

        x_train = self._fit_transform(x)
        rng = np.random.default_rng(self.random_seed)
        self.weights_ = rng.normal(scale=0.01, size=x_train.shape[1])
        self.bias_ = 0.0

        for _ in range(self.epochs):
            logits = x_train @ self.weights_ + self.bias_
            probs = _sigmoid(logits)
            error = probs - y

            grad_w = (x_train.T @ error) / len(y) + self.l2 * self.weights_
            grad_b = float(np.mean(error))

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Classifier must be fit before prediction.")
        x = self._transform(_as_2d_float(features))
        return _sigmoid(x @ self.weights_ + self.bias_)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(np.int64)

    def _fit_transform(self, features: np.ndarray) -> np.ndarray:
        if not self.standardize:
            self.mean_ = np.zeros(features.shape[1])
            self.scale_ = np.ones(features.shape[1])
            return features

        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return (features - self.mean_) / self.scale_

    def _transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Classifier standardization statistics are missing.")
        return (features - self.mean_) / self.scale_


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _as_2d_float(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Features contain non-finite values.")
    return x

