from __future__ import annotations

import numpy as np


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y, p = _validate(labels, scores)
    predictions = (p >= threshold).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (y == 1)))
    fp = int(np.sum((predictions == 1) & (y == 0)))
    tn = int(np.sum((predictions == 0) & (y == 0)))
    fn = int(np.sum((predictions == 0) & (y == 1)))
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    return {
        "n": float(len(y)),
        "prevalence": float(np.mean(y)),
        "threshold": float(threshold),
        "auroc": roc_auc(y, p),
        "auprc": average_precision(y, p),
        "precision": precision,
        "recall": recall,
        "f1": _divide(2.0 * precision * recall, precision + recall),
        "false_positive_rate": _divide(fp, fp + tn),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "true_negatives": float(tn),
        "false_negatives": float(fn),
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y, p = _validate(labels, scores)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_scores = p[order]
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y, p = _validate(labels, scores)
    positives = int(np.sum(y == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ranked = y[order]
    cumulative = np.cumsum(ranked)
    precision_at_rank = cumulative / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision_at_rank * ranked) / positives)


def _validate(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and scores must have the same non-zero length")
    if set(np.unique(y)) - {0, 1} or not np.all(np.isfinite(p)):
        raise ValueError("labels must be binary and scores finite")
    return y, p


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
