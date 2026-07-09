from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from AEGIS.labels import coerce_binary_labels
from AEGIS.metrics import detection_report


def summarize_decisions(
    decisions: pd.DataFrame,
    *,
    label_column: str | None = None,
    baseline_path: str | Path | None = None,
) -> dict[str, object]:
    required = {"action", "verdict", "risk_score", "uncertain"}
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"Decision table missing required columns: {missing}")

    scores = pd.to_numeric(decisions["risk_score"], errors="coerce").to_numpy(dtype=float)
    finite_scores = scores[np.isfinite(scores)]
    summary: dict[str, object] = {
        "n_samples": int(len(decisions)),
        "action_counts": _counts(decisions["action"]),
        "verdict_counts": _counts(decisions["verdict"]),
        "uncertain_rate": float(decisions["uncertain"].astype(bool).mean()),
        "score_summary": _score_summary(finite_scores),
    }

    if label_column and label_column in decisions.columns:
        labels = coerce_binary_labels(decisions[label_column])
        predictions = decisions["verdict"].astype(str).eq("malicious").astype(int).to_numpy()
        summary["labeled_metrics"] = detection_report(labels, np.nan_to_num(scores), predictions)

    if baseline_path is not None:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        summary["drift"] = drift_report(summary, baseline)

    return summary


def drift_report(current: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    current_scores = current.get("score_summary", {})
    baseline_scores = baseline.get("score_summary", {})
    action_delta = _rate_delta(
        current.get("action_counts", {}),
        baseline.get("action_counts", {}),
        int(current.get("n_samples", 0)),
        int(baseline.get("n_samples", 0)),
    )
    return {
        "action_rate_delta": action_delta,
        "mean_score_delta": _float_or_none(current_scores.get("mean"))
        - _float_or_none(baseline_scores.get("mean")),
        "p95_score_delta": _float_or_none(current_scores.get("p95"))
        - _float_or_none(baseline_scores.get("p95")),
        "uncertain_rate_delta": _float_or_none(current.get("uncertain_rate"))
        - _float_or_none(baseline.get("uncertain_rate")),
    }


def _score_summary(scores: np.ndarray) -> dict[str, float | None]:
    if len(scores) == 0:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(scores)),
        "p50": float(np.quantile(scores, 0.50)),
        "p95": float(np.quantile(scores, 0.95)),
        "max": float(np.max(scores)),
    }


def _counts(values: pd.Series) -> dict[str, int]:
    counts = values.astype(str).value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _rate_delta(
    current_counts: dict,
    baseline_counts: dict,
    current_n: int,
    baseline_n: int,
) -> dict[str, float]:
    keys = sorted(set(current_counts) | set(baseline_counts))
    output = {}
    for key in keys:
        current_rate = float(current_counts.get(key, 0)) / current_n if current_n else 0.0
        baseline_rate = float(baseline_counts.get(key, 0)) / baseline_n if baseline_n else 0.0
        output[str(key)] = current_rate - baseline_rate
    return output


def _float_or_none(value) -> float:
    return 0.0 if value is None else float(value)
