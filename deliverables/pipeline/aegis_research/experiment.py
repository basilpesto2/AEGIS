from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import FeatureBundle
from .metrics import classification_metrics
from .model import LogisticDetector, select_threshold
from .signals import binary_entropy, build_feature_views


@dataclass(frozen=True)
class ExperimentConfig:
    learning_rate: float = 0.05
    epochs: int = 800
    l2: float = 1e-3
    seed: int = 42
    trusted_per_class: int | None = None
    pseudo_per_class: int = 0
    pseudo_low: float = 0.15
    pseudo_high: float = 0.85


def run_ablation_suite(
    metadata: list[dict[str, str]],
    bundle: FeatureBundle,
    config: ExperimentConfig,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    labels = np.asarray([int(row["label_id"]) for row in metadata], dtype=np.int64)
    splits = np.asarray([row["split"] for row in metadata])
    views = build_feature_views(bundle.text_embeddings, bundle.image_embeddings, bundle.attribution_features)
    results: list[dict[str, object]] = []
    fitted: dict[str, dict[str, object]] = {}
    for offset, (name, features) in enumerate(views.items()):
        outcome = fit_evaluate_view(features, labels, splits, config, seed_offset=offset)
        result = {
            "feature_view": name,
            "feature_dim": int(features.shape[1]),
            "feature_source": bundle.feature_source,
            **outcome["test_metrics"],
            "validation_auroc": outcome["validation_metrics"]["auroc"],
            "validation_auprc": outcome["validation_metrics"]["auprc"],
            "mean_test_uncertainty": outcome["mean_test_uncertainty"],
            "n_trusted": outcome["n_trusted"],
            "n_pseudo": outcome["n_pseudo"],
        }
        results.append(result)
        fitted[name] = outcome
    return results, fitted


def fit_evaluate_view(
    features: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    config: ExperimentConfig,
    *,
    seed_offset: int = 0,
) -> dict[str, object]:
    train_idx = np.flatnonzero(splits == "train")
    validation_idx = np.flatnonzero(splits == "validation")
    test_idx = np.flatnonzero(splits == "test")
    if min(len(train_idx), len(validation_idx), len(test_idx)) == 0:
        raise ValueError("train, validation, and test splits must all be non-empty")
    rng = np.random.default_rng(config.seed + seed_offset)
    trusted_idx = _trusted_indices(train_idx, labels, config.trusted_per_class, rng)
    detector = _new_detector(config, seed_offset).fit(features[trusted_idx], labels[trusted_idx])
    pseudo_idx = np.asarray([], dtype=np.int64)
    pseudo_labels = np.asarray([], dtype=np.int64)
    if config.pseudo_per_class > 0:
        remaining = np.setdiff1d(train_idx, trusted_idx)
        scores = detector.predict_proba(features[remaining])
        pseudo_idx, pseudo_labels = _balanced_pseudo_indices(
            remaining,
            scores,
            config.pseudo_per_class,
            config.pseudo_low,
            config.pseudo_high,
        )
        if len(pseudo_idx):
            fit_idx = np.concatenate([trusted_idx, pseudo_idx])
            fit_labels = np.concatenate([labels[trusted_idx], pseudo_labels])
            detector = _new_detector(config, seed_offset).fit(features[fit_idx], fit_labels)

    validation_scores = detector.predict_proba(features[validation_idx])
    threshold, validation_metrics = select_threshold(labels[validation_idx], validation_scores)
    test_scores = detector.predict_proba(features[test_idx])
    test_metrics = classification_metrics(labels[test_idx], test_scores, threshold)
    return {
        "detector": detector,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_scores": test_scores,
        "test_indices": test_idx,
        "mean_test_uncertainty": float(binary_entropy(test_scores).mean()),
        "n_trusted": int(len(trusted_idx)),
        "n_pseudo": int(len(pseudo_idx)),
    }


def _new_detector(config: ExperimentConfig, offset: int) -> LogisticDetector:
    return LogisticDetector(
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        l2=config.l2,
        seed=config.seed + offset,
    )


def _trusted_indices(
    train_idx: np.ndarray,
    labels: np.ndarray,
    per_class: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if per_class is None:
        return train_idx
    selected: list[int] = []
    for label in (0, 1):
        candidates = train_idx[labels[train_idx] == label].copy()
        rng.shuffle(candidates)
        if len(candidates) < per_class:
            raise ValueError(f"requested {per_class} trusted rows for class {label}, only {len(candidates)} available")
        selected.extend(int(value) for value in candidates[:per_class])
    return np.asarray(sorted(selected), dtype=np.int64)


def _balanced_pseudo_indices(
    candidates: np.ndarray,
    scores: np.ndarray,
    per_class: int,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    benign_order = np.argsort(scores)
    malicious_order = np.argsort(-scores)
    benign = [int(candidates[i]) for i in benign_order if scores[i] <= low][:per_class]
    malicious = [int(candidates[i]) for i in malicious_order if scores[i] >= high][:per_class]
    count = min(len(benign), len(malicious))
    selected = np.asarray(benign[:count] + malicious[:count], dtype=np.int64)
    pseudo = np.asarray([0] * count + [1] * count, dtype=np.int64)
    return selected, pseudo
