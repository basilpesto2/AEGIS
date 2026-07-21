from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from aegis_research.metrics import average_precision, classification_metrics, roc_auc  # noqa: E402
from aegis_research.model import LogisticDetector, select_threshold  # noqa: E402
from aegis_research.signals import binary_entropy, cross_modal_consistency  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_perfect_ranking(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc(labels, scores), 1.0)
        self.assertAlmostEqual(average_precision(labels, scores), 1.0)
        self.assertEqual(classification_metrics(labels, scores, 0.5)["f1"], 1.0)

    def test_threshold_uses_supplied_validation_rows(self) -> None:
        threshold, metrics = select_threshold(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.4, 0.6, 0.9]))
        self.assertGreater(threshold, 0.4)
        self.assertEqual(metrics["f1"], 1.0)


class SignalTests(unittest.TestCase):
    def test_consistency_shape(self) -> None:
        text = np.eye(3)
        features, names = cross_modal_consistency(text, text.copy())
        self.assertEqual(features.shape, (3, 8))
        self.assertEqual(len(names), 8)
        self.assertTrue(np.allclose(features[:, 0], 1.0))

    def test_entropy_extremes(self) -> None:
        values = binary_entropy(np.asarray([0.0, 0.5, 1.0]))
        self.assertAlmostEqual(values[1], 1.0)
        self.assertLess(values[0], 1e-8)
        self.assertLess(values[2], 1e-8)


class DetectorTests(unittest.TestCase):
    def test_learns_separable_data(self) -> None:
        x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.asarray([0, 0, 1, 1])
        detector = LogisticDetector(epochs=500).fit(x, y)
        predictions = detector.predict_proba(x)
        self.assertTrue(np.all(predictions[:2] < 0.5))
        self.assertTrue(np.all(predictions[2:] > 0.5))


if __name__ == "__main__":
    unittest.main()
