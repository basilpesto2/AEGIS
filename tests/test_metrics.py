from __future__ import annotations

import numpy as np
import pytest

from AEGIS.metrics import average_precision, detection_report, fpr_at_tpr, roc_auc


def test_ranking_metrics_match_known_example() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.4, 0.35, 0.8])
    predictions = np.asarray([0, 1, 0, 1])

    report = detection_report(labels, scores, predictions)

    assert report["auroc"] == pytest.approx(0.75)
    assert report["auprc"] == pytest.approx(5 / 6)
    assert report["fpr95"] == pytest.approx(0.5)
    assert report["precision"] == pytest.approx(0.5)
    assert report["recall"] == pytest.approx(0.5)
    assert report["f1"] == pytest.approx(0.5)


def test_metrics_handle_tied_scores_with_average_ranks() -> None:
    labels = np.asarray([0, 1, 0, 1])
    scores = np.asarray([0.5, 0.5, 0.2, 0.8])

    assert roc_auc(labels, scores) == pytest.approx(0.875)
    assert average_precision(labels, scores) == pytest.approx((1.0 + 2 / 3) / 2)


def test_metrics_reject_single_class_inputs() -> None:
    with pytest.raises(ValueError, match="at least one positive and one negative"):
        fpr_at_tpr(np.asarray([1, 1, 1]), np.asarray([0.2, 0.4, 0.6]))
