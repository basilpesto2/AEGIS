from __future__ import annotations

import numpy as np


def _validate(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and scores must be 1D arrays.")
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same length.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1.")
    if np.sum(labels == 1) == 0 or np.sum(labels == 0) == 0:
        raise ValueError("labels must include at least one positive and one negative sample.")
    return labels, scores


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC with average ranks for tied scores."""
    labels, scores = _validate(labels, scores)
    order = np.argsort(scores)
    sorted_scores = scores[order]

    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    n = len(sorted_scores)
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[start:end] = average_rank
        start = end

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    rank_sum_pos = np.sum(original_ranks[labels == 1])
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute average precision for higher-is-more-malicious scores."""
    labels, scores = _validate(labels, scores)
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, len(labels) + 1)
    precision_at_k = cumulative_tp / ranks
    n_pos = np.sum(labels == 1)
    return float(np.sum(precision_at_k[sorted_labels == 1]) / n_pos)


def fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    labels, scores = _validate(labels, scores)
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    tpr = tp / n_pos
    fpr = fp / n_neg
    valid = np.where(tpr >= target_tpr)[0]
    if len(valid) == 0:
        return 1.0
    return float(np.min(fpr[valid]))


def precision_recall_f1(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape.")

    tp = np.sum((labels == 1) & (predictions == 1))
    fp = np.sum((labels == 0) & (predictions == 1))
    fn = np.sum((labels == 1) & (predictions == 0))
    tn = np.sum((labels == 0) & (predictions == 0))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def detection_report(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray | None = None,
) -> dict[str, float]:
    report = {
        "auroc": roc_auc(labels, scores),
        "auprc": average_precision(labels, scores),
        "fpr95": fpr_at_tpr(labels, scores, target_tpr=0.95),
    }
    if predictions is not None:
        report.update(precision_recall_f1(labels, predictions))
    return report

