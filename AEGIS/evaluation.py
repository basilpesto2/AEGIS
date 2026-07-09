from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from AEGIS.logistic import LogisticRegressionNumpy
from AEGIS.metrics import detection_report
from AEGIS.svd_detector import SVDMaliciousnessDetector, pseudo_labels_from_scores


@dataclass(frozen=True)
class SVDAblationResult:
    n_components: int
    orientation: str
    n_fit_samples: int
    n_eval_samples: int
    predicted_positive_rate: float | None
    metrics: dict[str, float]

    def as_row(self) -> dict[str, float | int | str | None]:
        row: dict[str, float | int | str | None] = {
            "n_components": self.n_components,
            "orientation": self.orientation,
            "n_fit_samples": self.n_fit_samples,
            "n_eval_samples": self.n_eval_samples,
            "predicted_positive_rate": self.predicted_positive_rate,
        }
        row.update(self.metrics)
        return row


@dataclass(frozen=True)
class SVDValidationResult:
    n_components: int
    orientation: str
    n_fit_samples: int
    n_validation_samples: int
    n_test_samples: int
    threshold: float | None
    validation_positive_rate: float | None
    test_positive_rate: float | None
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]

    def as_row(self) -> dict[str, float | int | str | None]:
        row: dict[str, float | int | str | None] = {
            "n_components": self.n_components,
            "orientation": self.orientation,
            "n_fit_samples": self.n_fit_samples,
            "n_validation_samples": self.n_validation_samples,
            "n_test_samples": self.n_test_samples,
            "threshold": self.threshold,
            "validation_positive_rate": self.validation_positive_rate,
            "test_positive_rate": self.test_positive_rate,
        }
        row.update({f"validation_{key}": value for key, value in self.validation_metrics.items()})
        row.update({f"test_{key}": value for key, value in self.test_metrics.items()})
        return row


@dataclass(frozen=True)
class LogisticValidationResult:
    learning_rate: float
    l2: float
    epochs: int
    n_fit_samples: int
    n_validation_samples: int
    n_test_samples: int
    threshold: float | None
    validation_positive_rate: float | None
    test_positive_rate: float | None
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]

    def as_row(self) -> dict[str, float | int | str | None]:
        row: dict[str, float | int | str | None] = {
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "epochs": self.epochs,
            "n_fit_samples": self.n_fit_samples,
            "n_validation_samples": self.n_validation_samples,
            "n_test_samples": self.n_test_samples,
            "threshold": self.threshold,
            "validation_positive_rate": self.validation_positive_rate,
            "test_positive_rate": self.test_positive_rate,
        }
        row.update({f"validation_{key}": value for key, value in self.validation_metrics.items()})
        row.update({f"test_{key}": value for key, value in self.test_metrics.items()})
        return row


@dataclass(frozen=True)
class LogisticTransferResult:
    learning_rate: float
    l2: float
    epochs: int
    n_source_fit_samples: int
    n_source_validation_samples: int
    n_source_test_samples: int
    n_target_samples: int
    threshold: float | None
    source_validation_positive_rate: float | None
    source_test_positive_rate: float | None
    target_positive_rate: float | None
    source_validation_metrics: dict[str, float]
    source_test_metrics: dict[str, float]
    target_metrics: dict[str, float]

    def as_row(self) -> dict[str, float | int | str | None]:
        row: dict[str, float | int | str | None] = {
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "epochs": self.epochs,
            "n_source_fit_samples": self.n_source_fit_samples,
            "n_source_validation_samples": self.n_source_validation_samples,
            "n_source_test_samples": self.n_source_test_samples,
            "n_target_samples": self.n_target_samples,
            "threshold": self.threshold,
            "source_validation_positive_rate": self.source_validation_positive_rate,
            "source_test_positive_rate": self.source_test_positive_rate,
            "target_positive_rate": self.target_positive_rate,
        }
        row.update(
            {f"source_validation_{key}": value for key, value in self.source_validation_metrics.items()}
        )
        row.update({f"source_test_{key}": value for key, value in self.source_test_metrics.items()})
        row.update({f"target_{key}": value for key, value in self.target_metrics.items()})
        return row


@dataclass(frozen=True)
class PseudoLabelLogisticValidationResult:
    svd_components: int
    orientation: str
    benign_keep_fraction: float | None
    learning_rate: float
    l2: float
    epochs: int
    n_fit_samples: int
    n_pseudo_samples: int
    n_pseudo_malicious: int
    n_pseudo_benign: int
    pseudo_label_accuracy: float | None
    pseudo_malicious_precision: float | None
    pseudo_benign_precision: float | None
    n_validation_samples: int
    n_test_samples: int
    threshold: float | None
    validation_positive_rate: float | None
    test_positive_rate: float | None
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]

    def as_row(self) -> dict[str, float | int | str | None]:
        row: dict[str, float | int | str | None] = {
            "svd_components": self.svd_components,
            "orientation": self.orientation,
            "benign_keep_fraction": self.benign_keep_fraction,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "epochs": self.epochs,
            "n_fit_samples": self.n_fit_samples,
            "n_pseudo_samples": self.n_pseudo_samples,
            "n_pseudo_malicious": self.n_pseudo_malicious,
            "n_pseudo_benign": self.n_pseudo_benign,
            "pseudo_label_accuracy": self.pseudo_label_accuracy,
            "pseudo_malicious_precision": self.pseudo_malicious_precision,
            "pseudo_benign_precision": self.pseudo_benign_precision,
            "n_validation_samples": self.n_validation_samples,
            "n_test_samples": self.n_test_samples,
            "threshold": self.threshold,
            "validation_positive_rate": self.validation_positive_rate,
            "test_positive_rate": self.test_positive_rate,
        }
        row.update({f"validation_{key}": value for key, value in self.validation_metrics.items()})
        row.update({f"test_{key}": value for key, value in self.test_metrics.items()})
        return row


def orient_scores(scores: np.ndarray, orientation: str) -> np.ndarray:
    """Return scores where larger values mean more malicious."""
    scores = np.asarray(scores, dtype=np.float64)
    if orientation == "normal":
        return scores
    if orientation == "inverted":
        return -scores
    raise ValueError("orientation must be 'normal' or 'inverted'.")


def threshold_from_prior(scores: np.ndarray, malicious_prior: float) -> float:
    """Choose a score threshold from an expected malicious prior."""
    if not 0.0 < malicious_prior < 1.0:
        raise ValueError("malicious_prior must be between 0 and 1.")
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.quantile(scores, 1.0 - malicious_prior))


def predict_top_prior(scores: np.ndarray, malicious_prior: float) -> np.ndarray:
    """Predict the highest-scoring prior fraction as malicious."""
    scores = np.asarray(scores, dtype=np.float64)
    threshold = threshold_from_prior(scores, malicious_prior)
    return (scores >= threshold).astype(np.int64)


def predict_from_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int64)


def run_svd_component_grid(
    embeddings: np.ndarray,
    components: Iterable[int],
    labels: np.ndarray | None = None,
    fit_mask: np.ndarray | None = None,
    eval_mask: np.ndarray | None = None,
    orientations: Iterable[str] = ("normal", "inverted"),
    malicious_prior: float | None = None,
) -> list[SVDAblationResult]:
    """Evaluate SVD detector settings over component counts and score directions."""
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {x.shape}.")

    fit_mask = _mask_or_all(fit_mask, len(x), "fit_mask")
    eval_mask = _mask_or_all(eval_mask, len(x), "eval_mask")
    y = None if labels is None else np.asarray(labels, dtype=np.int64)
    if y is not None and y.shape[0] != len(x):
        raise ValueError("labels must have the same number of rows as embeddings.")

    results: list[SVDAblationResult] = []
    for n_components in components:
        detector = SVDMaliciousnessDetector(n_components=int(n_components)).fit(x[fit_mask])
        raw_scores = detector.score_samples(x)

        for orientation in orientations:
            oriented = orient_scores(raw_scores, orientation)
            eval_scores = oriented[eval_mask]
            predictions = None
            positive_rate = None
            if malicious_prior is not None:
                predictions = predict_top_prior(eval_scores, malicious_prior)
                positive_rate = float(np.mean(predictions))

            metrics: dict[str, float] = {}
            if y is not None:
                metrics = detection_report(y[eval_mask], eval_scores, predictions)

            results.append(
                SVDAblationResult(
                    n_components=int(n_components),
                    orientation=orientation,
                    n_fit_samples=int(np.sum(fit_mask)),
                    n_eval_samples=int(np.sum(eval_mask)),
                    predicted_positive_rate=positive_rate,
                    metrics=metrics,
                )
            )
    return results


def run_svd_validation_grid(
    embeddings: np.ndarray,
    components: Iterable[int],
    labels: np.ndarray,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    orientations: Iterable[str] = ("normal", "inverted"),
    malicious_prior: float | None = None,
) -> list[SVDValidationResult]:
    """Evaluate SVD settings with validation-thresholded test reporting."""
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {x.shape}.")

    y = np.asarray(labels, dtype=np.int64)
    if y.shape[0] != len(x):
        raise ValueError("labels must have the same number of rows as embeddings.")

    fit_mask = _mask_or_all(fit_mask, len(x), "fit_mask")
    validation_mask = _mask_or_all(validation_mask, len(x), "validation_mask")
    test_mask = _mask_or_all(test_mask, len(x), "test_mask")

    results: list[SVDValidationResult] = []
    for n_components in components:
        detector = SVDMaliciousnessDetector(n_components=int(n_components)).fit(x[fit_mask])
        raw_scores = detector.score_samples(x)

        for orientation in orientations:
            oriented = orient_scores(raw_scores, orientation)
            validation_scores = oriented[validation_mask]
            test_scores = oriented[test_mask]

            threshold = None
            validation_predictions = None
            test_predictions = None
            validation_positive_rate = None
            test_positive_rate = None
            if malicious_prior is not None:
                threshold = threshold_from_prior(validation_scores, malicious_prior)
                validation_predictions = predict_from_threshold(validation_scores, threshold)
                test_predictions = predict_from_threshold(test_scores, threshold)
                validation_positive_rate = float(np.mean(validation_predictions))
                test_positive_rate = float(np.mean(test_predictions))

            results.append(
                SVDValidationResult(
                    n_components=int(n_components),
                    orientation=orientation,
                    n_fit_samples=int(np.sum(fit_mask)),
                    n_validation_samples=int(np.sum(validation_mask)),
                    n_test_samples=int(np.sum(test_mask)),
                    threshold=threshold,
                    validation_positive_rate=validation_positive_rate,
                    test_positive_rate=test_positive_rate,
                    validation_metrics=detection_report(
                        y[validation_mask],
                        validation_scores,
                        validation_predictions,
                    ),
                    test_metrics=detection_report(
                        y[test_mask],
                        test_scores,
                        test_predictions,
                    ),
                )
            )
    return results


def run_logistic_validation_grid(
    features: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    learning_rates: Iterable[float] = (0.03, 0.1),
    l2_values: Iterable[float] = (1e-4,),
    epochs_values: Iterable[int] = (500,),
    malicious_prior: float | None = None,
    random_seed: int = 42,
) -> list[LogisticValidationResult]:
    """Train supervised logistic classifiers and report validation/test metrics."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {x.shape}.")

    y = np.asarray(labels, dtype=np.int64)
    if y.shape[0] != len(x):
        raise ValueError("labels must have the same number of rows as features.")

    fit_mask = _mask_or_all(fit_mask, len(x), "fit_mask")
    validation_mask = _mask_or_all(validation_mask, len(x), "validation_mask")
    test_mask = _mask_or_all(test_mask, len(x), "test_mask")

    results: list[LogisticValidationResult] = []
    for learning_rate in learning_rates:
        for l2 in l2_values:
            for epochs in epochs_values:
                classifier = LogisticRegressionNumpy(
                    learning_rate=float(learning_rate),
                    l2=float(l2),
                    epochs=int(epochs),
                    random_seed=random_seed,
                ).fit(x[fit_mask], y[fit_mask])

                validation_scores = classifier.predict_proba(x[validation_mask])
                test_scores = classifier.predict_proba(x[test_mask])
                threshold = None
                validation_predictions = None
                test_predictions = None
                validation_positive_rate = None
                test_positive_rate = None
                if malicious_prior is not None:
                    threshold = threshold_from_prior(validation_scores, malicious_prior)
                    validation_predictions = predict_from_threshold(validation_scores, threshold)
                    test_predictions = predict_from_threshold(test_scores, threshold)
                    validation_positive_rate = float(np.mean(validation_predictions))
                    test_positive_rate = float(np.mean(test_predictions))

                results.append(
                    LogisticValidationResult(
                        learning_rate=float(learning_rate),
                        l2=float(l2),
                        epochs=int(epochs),
                        n_fit_samples=int(np.sum(fit_mask)),
                        n_validation_samples=int(np.sum(validation_mask)),
                        n_test_samples=int(np.sum(test_mask)),
                        threshold=threshold,
                        validation_positive_rate=validation_positive_rate,
                        test_positive_rate=test_positive_rate,
                        validation_metrics=detection_report(
                            y[validation_mask],
                            validation_scores,
                            validation_predictions,
                        ),
                        test_metrics=detection_report(
                            y[test_mask],
                            test_scores,
                            test_predictions,
                        ),
                    )
                )
    return results


def run_logistic_transfer_grid(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    source_fit_mask: np.ndarray,
    source_validation_mask: np.ndarray,
    source_test_mask: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    learning_rates: Iterable[float] = (0.03, 0.1),
    l2_values: Iterable[float] = (1e-4,),
    epochs_values: Iterable[int] = (500,),
    malicious_prior: float | None = None,
    random_seed: int = 42,
) -> list[LogisticTransferResult]:
    """Train on source features and evaluate transfer on a separate target dataset."""
    source_x = np.asarray(source_features, dtype=np.float64)
    target_x = np.asarray(target_features, dtype=np.float64)
    if source_x.ndim != 2:
        raise ValueError(f"source_features must be 2D, got shape {source_x.shape}.")
    if target_x.ndim != 2:
        raise ValueError(f"target_features must be 2D, got shape {target_x.shape}.")
    if source_x.shape[1] != target_x.shape[1]:
        raise ValueError(
            "source_features and target_features must have the same feature dimension."
        )

    source_y = np.asarray(source_labels, dtype=np.int64)
    target_y = np.asarray(target_labels, dtype=np.int64)
    if source_y.shape[0] != len(source_x):
        raise ValueError("source_labels must have the same number of rows as source_features.")
    if target_y.shape[0] != len(target_x):
        raise ValueError("target_labels must have the same number of rows as target_features.")

    source_fit_mask = _mask_or_all(source_fit_mask, len(source_x), "source_fit_mask")
    source_validation_mask = _mask_or_all(
        source_validation_mask,
        len(source_x),
        "source_validation_mask",
    )
    source_test_mask = _mask_or_all(source_test_mask, len(source_x), "source_test_mask")

    results: list[LogisticTransferResult] = []
    for learning_rate in learning_rates:
        for l2 in l2_values:
            for epochs in epochs_values:
                classifier = LogisticRegressionNumpy(
                    learning_rate=float(learning_rate),
                    l2=float(l2),
                    epochs=int(epochs),
                    random_seed=random_seed,
                ).fit(source_x[source_fit_mask], source_y[source_fit_mask])

                source_validation_scores = classifier.predict_proba(
                    source_x[source_validation_mask]
                )
                source_test_scores = classifier.predict_proba(source_x[source_test_mask])
                target_scores = classifier.predict_proba(target_x)

                threshold = None
                source_validation_predictions = None
                source_test_predictions = None
                target_predictions = None
                source_validation_positive_rate = None
                source_test_positive_rate = None
                target_positive_rate = None
                if malicious_prior is not None:
                    threshold = threshold_from_prior(source_validation_scores, malicious_prior)
                    source_validation_predictions = predict_from_threshold(
                        source_validation_scores,
                        threshold,
                    )
                    source_test_predictions = predict_from_threshold(
                        source_test_scores,
                        threshold,
                    )
                    target_predictions = predict_from_threshold(target_scores, threshold)
                    source_validation_positive_rate = float(
                        np.mean(source_validation_predictions)
                    )
                    source_test_positive_rate = float(np.mean(source_test_predictions))
                    target_positive_rate = float(np.mean(target_predictions))

                results.append(
                    LogisticTransferResult(
                        learning_rate=float(learning_rate),
                        l2=float(l2),
                        epochs=int(epochs),
                        n_source_fit_samples=int(np.sum(source_fit_mask)),
                        n_source_validation_samples=int(np.sum(source_validation_mask)),
                        n_source_test_samples=int(np.sum(source_test_mask)),
                        n_target_samples=int(len(target_x)),
                        threshold=threshold,
                        source_validation_positive_rate=source_validation_positive_rate,
                        source_test_positive_rate=source_test_positive_rate,
                        target_positive_rate=target_positive_rate,
                        source_validation_metrics=detection_report(
                            source_y[source_validation_mask],
                            source_validation_scores,
                            source_validation_predictions,
                        ),
                        source_test_metrics=detection_report(
                            source_y[source_test_mask],
                            source_test_scores,
                            source_test_predictions,
                        ),
                        target_metrics=detection_report(
                            target_y,
                            target_scores,
                            target_predictions,
                        ),
                    )
                )
    return results


def run_pseudo_label_logistic_validation_grid(
    features: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    svd_components: Iterable[int],
    orientations: Iterable[str] = ("normal", "inverted"),
    malicious_prior: float = 0.167,
    benign_keep_fractions: Iterable[float | None] = (None,),
    learning_rates: Iterable[float] = (0.03, 0.1),
    l2_values: Iterable[float] = (1e-4,),
    epochs_values: Iterable[int] = (500,),
    random_seed: int = 42,
) -> list[PseudoLabelLogisticValidationResult]:
    """Train logistic classifiers from SVD pseudo-labels and report validation/test metrics."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {x.shape}.")

    y = np.asarray(labels, dtype=np.int64)
    if y.shape[0] != len(x):
        raise ValueError("labels must have the same number of rows as features.")

    fit_mask = _mask_or_all(fit_mask, len(x), "fit_mask")
    validation_mask = _mask_or_all(validation_mask, len(x), "validation_mask")
    test_mask = _mask_or_all(test_mask, len(x), "test_mask")
    fit_indices = np.where(fit_mask)[0]

    results: list[PseudoLabelLogisticValidationResult] = []
    for n_components in svd_components:
        detector = SVDMaliciousnessDetector(n_components=int(n_components)).fit(x[fit_mask])
        raw_fit_scores = detector.score_samples(x[fit_mask])

        for orientation in orientations:
            oriented_fit_scores = orient_scores(raw_fit_scores, orientation)
            for benign_keep_fraction in benign_keep_fractions:
                pseudo_local_indices, pseudo_labels = pseudo_labels_from_scores(
                    oriented_fit_scores,
                    malicious_prior=malicious_prior,
                    benign_keep_fraction=benign_keep_fraction,
                )
                pseudo_global_indices = fit_indices[pseudo_local_indices]

                diagnostics = _pseudo_label_diagnostics(
                    y[pseudo_global_indices],
                    pseudo_labels,
                )

                for learning_rate in learning_rates:
                    for l2 in l2_values:
                        for epochs in epochs_values:
                            classifier = LogisticRegressionNumpy(
                                learning_rate=float(learning_rate),
                                l2=float(l2),
                                epochs=int(epochs),
                                random_seed=random_seed,
                            ).fit(x[pseudo_global_indices], pseudo_labels)

                            validation_scores = classifier.predict_proba(x[validation_mask])
                            test_scores = classifier.predict_proba(x[test_mask])
                            threshold = threshold_from_prior(validation_scores, malicious_prior)
                            validation_predictions = predict_from_threshold(
                                validation_scores,
                                threshold,
                            )
                            test_predictions = predict_from_threshold(test_scores, threshold)

                            results.append(
                                PseudoLabelLogisticValidationResult(
                                    svd_components=int(n_components),
                                    orientation=orientation,
                                    benign_keep_fraction=benign_keep_fraction,
                                    learning_rate=float(learning_rate),
                                    l2=float(l2),
                                    epochs=int(epochs),
                                    n_fit_samples=int(np.sum(fit_mask)),
                                    n_pseudo_samples=int(len(pseudo_labels)),
                                    n_pseudo_malicious=int(np.sum(pseudo_labels == 1)),
                                    n_pseudo_benign=int(np.sum(pseudo_labels == 0)),
                                    pseudo_label_accuracy=diagnostics["accuracy"],
                                    pseudo_malicious_precision=diagnostics["malicious_precision"],
                                    pseudo_benign_precision=diagnostics["benign_precision"],
                                    n_validation_samples=int(np.sum(validation_mask)),
                                    n_test_samples=int(np.sum(test_mask)),
                                    threshold=threshold,
                                    validation_positive_rate=float(np.mean(validation_predictions)),
                                    test_positive_rate=float(np.mean(test_predictions)),
                                    validation_metrics=detection_report(
                                        y[validation_mask],
                                        validation_scores,
                                        validation_predictions,
                                    ),
                                    test_metrics=detection_report(
                                        y[test_mask],
                                        test_scores,
                                        test_predictions,
                                    ),
                                )
                            )
    return results


def _pseudo_label_diagnostics(
    true_labels: np.ndarray,
    pseudo_labels: np.ndarray,
) -> dict[str, float | None]:
    true_labels = np.asarray(true_labels, dtype=np.int64)
    pseudo_labels = np.asarray(pseudo_labels, dtype=np.int64)
    diagnostics: dict[str, float | None] = {
        "accuracy": float(np.mean(true_labels == pseudo_labels)),
        "malicious_precision": None,
        "benign_precision": None,
    }
    malicious_mask = pseudo_labels == 1
    if np.any(malicious_mask):
        diagnostics["malicious_precision"] = float(np.mean(true_labels[malicious_mask] == 1))
    benign_mask = pseudo_labels == 0
    if np.any(benign_mask):
        diagnostics["benign_precision"] = float(np.mean(true_labels[benign_mask] == 0))
    return diagnostics


def _mask_or_all(mask: np.ndarray | None, n_rows: int, name: str) -> np.ndarray:
    if mask is None:
        return np.ones(n_rows, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (n_rows,):
        raise ValueError(f"{name} must be a boolean vector of length {n_rows}.")
    if not np.any(mask):
        raise ValueError(f"{name} selects no rows.")
    return mask
