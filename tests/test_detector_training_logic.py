from __future__ import annotations

import csv
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from AEGIS.logistic import LogisticRegressionNumpy


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "deliverables" / "pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from aegis_research import bordair_training as TRAINING  # noqa: E402
from aegis_research import bordair_corpus as CORPUS  # noqa: E402


class Schema4CorpusTransitionTests(unittest.TestCase):
    def test_active_hard_pair_and_historical_rows_are_protected(self) -> None:
        archived = [
            {"sample_id": "active-hard", "hard_negative": "1"},
            {"sample_id": "historical-ordinary", "hard_negative": "0"},
            {"sample_id": "pair-ordinary", "hard_negative": "0"},
            {"sample_id": "ordinary", "hard_negative": "0"},
        ]
        protected = CORPUS.schema4_protected_train_ids(
            archived,
            ["pair-ordinary"],
            {"historical-ordinary"},
        )
        self.assertEqual(
            protected,
            {"active-hard", "historical-ordinary", "pair-ordinary"},
        )

    def _anchored_schema4_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        manifest = root / "corpus_manifest_v7.json"
        development = root / "development_metadata_v7.csv"
        images = root / "images_v7"
        images.mkdir()
        manifest.write_text('{"schema_version":4}', encoding="utf-8")
        development.write_text("sample_id,label_id\nBTI-1,0\n", encoding="utf-8")
        (images / "BTI-1.png").write_bytes(b"pinned-image")
        return manifest, development, images

    def _anchor_patches(
        self,
        manifest: Path,
        development: Path,
        images: Path,
    ) -> tuple[patch, patch, patch]:
        return (
            patch.object(
                CORPUS,
                "EXPECTED_SCHEMA4_MANIFEST_SHA256",
                CORPUS.sha256(manifest),
            ),
            patch.object(
                CORPUS,
                "EXPECTED_SCHEMA4_DEVELOPMENT_SHA256",
                CORPUS.sha256(development),
            ),
            patch.object(
                CORPUS,
                "EXPECTED_SCHEMA4_IMAGE_TREE",
                CORPUS.tree_digest(images),
            ),
        )

    def test_schema4_anchor_rejects_development_metadata_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, development, images = self._anchored_schema4_fixture(
                Path(directory)
            )
            manifest_patch, development_patch, image_patch = self._anchor_patches(
                manifest,
                development,
                images,
            )
            with manifest_patch, development_patch, image_patch:
                CORPUS.validate_schema4_anchors(manifest, development, images)
                development.write_text(
                    "sample_id,label_id\nBTI-1,1\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "metadata hash changed"):
                    CORPUS.validate_schema4_anchors(manifest, development, images)

    def test_schema4_anchor_rejects_manifest_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, development, images = self._anchored_schema4_fixture(
                Path(directory)
            )
            manifest_patch, development_patch, image_patch = self._anchor_patches(
                manifest,
                development,
                images,
            )
            with manifest_patch, development_patch, image_patch:
                CORPUS.validate_schema4_anchors(manifest, development, images)
                manifest.write_text('{"schema_version":5}', encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "manifest hash changed"):
                    CORPUS.validate_schema4_anchors(manifest, development, images)

    def test_schema4_anchor_rejects_image_tree_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, development, images = self._anchored_schema4_fixture(
                Path(directory)
            )
            manifest_patch, development_patch, image_patch = self._anchor_patches(
                manifest,
                development,
                images,
            )
            with manifest_patch, development_patch, image_patch:
                CORPUS.validate_schema4_anchors(manifest, development, images)
                (images / "BTI-1.png").write_bytes(b"mutated-image")
                with self.assertRaisesRegex(RuntimeError, "image tree changed"):
                    CORPUS.validate_schema4_anchors(manifest, development, images)


class LogisticSampleWeightTests(unittest.TestCase):
    def test_uniform_weights_match_unweighted_fit(self) -> None:
        features = np.asarray(
            [[-2.0, 0.2], [-1.0, 0.4], [0.5, 0.7], [2.0, 1.2]],
            dtype=np.float64,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        settings = dict(learning_rate=0.05, epochs=120, l2=0.01, random_seed=7)
        plain = LogisticRegressionNumpy(**settings).fit(features, labels)
        weighted = LogisticRegressionNumpy(**settings).fit(
            features,
            labels,
            sample_weight=np.ones(len(labels)),
        )
        np.testing.assert_allclose(weighted.weights_, plain.weights_, atol=1e-12)
        np.testing.assert_allclose(weighted.mean_, plain.mean_, atol=1e-12)
        np.testing.assert_allclose(weighted.scale_, plain.scale_, atol=1e-12)
        self.assertAlmostEqual(weighted.bias_, plain.bias_, places=12)

    def test_integer_weights_match_row_duplication(self) -> None:
        features = np.asarray(
            [[-1.5, 0.0], [-0.5, 1.0], [0.5, 0.5], [1.5, 2.0]],
            dtype=np.float64,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        weights = np.asarray([1, 3, 2, 1], dtype=np.int64)
        settings = dict(learning_rate=0.03, epochs=150, l2=0.05, random_seed=11)
        weighted = LogisticRegressionNumpy(**settings).fit(
            features,
            labels,
            sample_weight=weights,
        )
        duplicated = LogisticRegressionNumpy(**settings).fit(
            np.repeat(features, weights, axis=0),
            np.repeat(labels, weights),
        )
        np.testing.assert_allclose(weighted.weights_, duplicated.weights_, atol=1e-12)
        np.testing.assert_allclose(weighted.mean_, duplicated.mean_, atol=1e-12)
        np.testing.assert_allclose(weighted.scale_, duplicated.scale_, atol=1e-12)
        self.assertAlmostEqual(weighted.bias_, duplicated.bias_, places=12)

    def test_invalid_weights_are_rejected(self) -> None:
        features = np.asarray([[0.0], [1.0]])
        labels = np.asarray([0, 1])
        for weights in (
            np.asarray([1.0]),
            np.asarray([1.0, -1.0]),
            np.asarray([1.0, np.nan]),
            np.asarray([0.0, 0.0]),
        ):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    LogisticRegressionNumpy(epochs=1).fit(
                        features,
                        labels,
                        sample_weight=weights,
                    )


class V7ThresholdAndWeightingTests(unittest.TestCase):
    def test_threshold_uses_predecessor_of_weakest_malicious_score(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.7])
        groups = {
            "all_malicious": np.asarray([False, False, True, True]),
            "image_only": np.asarray([False, False, True, False]),
            "text_led": np.asarray([False, False, False, True]),
        }
        threshold = TRAINING.select_block_threshold(labels, scores, groups)
        expected = float(
            np.nextafter(np.float64(0.7), np.float64(-np.inf))
        )
        self.assertEqual(threshold, expected)
        self.assertLess(threshold, 0.7)
        self.assertEqual(
            float(np.nextafter(threshold, np.float64(np.inf))),
            0.7,
        )
        self.assertTrue(np.all(scores[labels == 1] >= threshold))
        self.assertFalse(np.any(scores[labels == 0] >= threshold))

    def test_threshold_does_not_spend_available_false_positive_budget(self) -> None:
        benign = np.asarray([0.01] * 77 + [0.096326, 0.6169, 0.9516])
        malicious = np.asarray([0.424974, 0.8])
        labels = np.asarray([0] * len(benign) + [1, 1])
        scores = np.concatenate((benign, malicious))
        groups = {
            "all_malicious": labels == 1,
            "image_only": np.arange(len(labels)) == len(benign),
            "text_led": np.arange(len(labels)) == len(benign) + 1,
        }
        threshold = TRAINING.select_block_threshold(labels, scores, groups)
        self.assertEqual(
            threshold,
            float(
                np.nextafter(np.float64(0.424974), np.float64(-np.inf))
            ),
        )
        self.assertEqual(int(np.sum(benign >= threshold)), 2)
        self.assertTrue(np.all(malicious >= threshold))

    def test_fully_separated_scores_do_not_spend_false_positive_budget(self) -> None:
        benign = np.asarray([0.01] * 78 + [0.68, 0.69])
        malicious = np.asarray([0.7, 0.9])
        labels = np.asarray([0] * len(benign) + [1, 1])
        scores = np.concatenate((benign, malicious))
        groups = {
            "all_malicious": labels == 1,
            "image_only": np.arange(len(labels)) == len(benign),
            "text_led": np.arange(len(labels)) == len(benign) + 1,
        }
        threshold = TRAINING.select_block_threshold(labels, scores, groups)
        self.assertEqual(
            threshold,
            float(np.nextafter(np.float64(0.7), np.float64(-np.inf))),
        )
        self.assertEqual(int(np.sum(benign >= threshold)), 0)
        self.assertTrue(np.all(malicious >= threshold))

    def test_ties_at_recall_boundary_count_as_unavoidable_false_positives(
        self,
    ) -> None:
        benign = np.asarray([0.1] * 78 + [0.7, 0.7])
        malicious = np.asarray([0.7, 0.8])
        labels = np.asarray([0] * len(benign) + [1, 1])
        scores = np.concatenate((benign, malicious))
        groups = {
            "all_malicious": labels == 1,
            "image_only": np.arange(len(labels)) == len(benign),
            "text_led": np.arange(len(labels)) == len(benign) + 1,
        }
        threshold = TRAINING.select_block_threshold(labels, scores, groups)
        self.assertEqual(
            threshold,
            float(np.nextafter(np.float64(0.7), np.float64(-np.inf))),
        )
        self.assertEqual(int(np.sum(benign >= threshold)), 2)
        self.assertTrue(np.all(malicious >= threshold))

    def test_threshold_handles_fully_separated_probability_endpoints(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.0, 0.0, 1.0, 1.0])
        groups = {
            "all_malicious": labels == 1,
            "image_only": np.asarray([False, False, True, False]),
            "text_led": np.asarray([False, False, False, True]),
        }
        threshold = TRAINING.select_block_threshold(labels, scores, groups)
        self.assertEqual(
            threshold,
            float(np.nextafter(np.float64(1.0), np.float64(-np.inf))),
        )
        self.assertEqual(int(np.sum(scores[labels == 0] >= threshold)), 0)

    def test_false_positive_budget_is_floored_not_rounded(self) -> None:
        rate = 2.0 / 80.0
        self.assertEqual(TRAINING.false_positive_budget(39, rate), 0)
        self.assertEqual(TRAINING.false_positive_budget(79, rate), 1)
        self.assertEqual(TRAINING.false_positive_budget(80, rate), 2)

    def test_tied_benign_scores_never_exceed_budget_when_gap_exists(self) -> None:
        benign = np.asarray([0.1] * 77 + [0.6, 0.6, 0.6])
        malicious = np.asarray([0.7, 0.8])
        labels = np.asarray([0] * 80 + [1, 1])
        scores = np.concatenate((benign, malicious))
        groups = {
            "all_malicious": labels == 1,
            "image_only": np.arange(len(labels)) == 80,
            "text_led": np.arange(len(labels)) == 81,
        }
        threshold = TRAINING.select_block_threshold(labels, scores, groups)
        self.assertEqual(
            threshold,
            float(np.nextafter(np.float64(0.7), np.float64(-np.inf))),
        )
        self.assertEqual(int(np.sum(benign >= threshold)), 0)

    def test_no_gap_returns_recall_boundary_that_exposes_failed_budget(self) -> None:
        benign = np.asarray([0.1] * 77 + [0.7, 0.7, 0.7])
        malicious = np.asarray([0.7, 0.8])
        labels = np.asarray([0] * 80 + [1, 1])
        scores = np.concatenate((benign, malicious))
        groups = {
            "all_malicious": labels == 1,
            "image_only": np.arange(len(labels)) == 80,
            "text_led": np.arange(len(labels)) == 81,
        }
        threshold = TRAINING.select_block_threshold(labels, scores, groups)
        self.assertEqual(
            threshold,
            float(np.nextafter(np.float64(0.7), np.float64(-np.inf))),
        )
        self.assertTrue(np.all(malicious >= threshold))
        self.assertGreater(
            int(np.sum(benign >= threshold)),
            TRAINING.false_positive_budget(80, 2.0 / 80.0),
        )

    def test_zero_weakest_malicious_score_has_no_valid_artifact_threshold(
        self,
    ) -> None:
        labels = np.asarray([0, 1, 1])
        scores = np.asarray([0.0, 0.0, 0.8])
        groups = {
            "all_malicious": labels == 1,
            "image_only": np.asarray([False, True, False]),
            "text_led": np.asarray([False, False, True]),
        }
        with self.assertRaisesRegex(ValueError, "positive detector threshold"):
            TRAINING.select_block_threshold(labels, scores, groups)

    def test_operating_threshold_uses_lower_prior_regression_cap(self) -> None:
        validation_malicious = np.asarray([0.4585362448377539, 0.8])
        prior_regression_malicious = np.asarray([0.39317152135806194, 0.9])
        threshold = TRAINING.select_operating_block_threshold(
            validation_malicious,
            prior_regression_malicious,
        )
        expected = float(
            np.nextafter(
                np.float64(0.39317152135806194),
                np.float64(-np.inf),
            )
        )
        self.assertEqual(threshold, expected)
        self.assertTrue(np.all(validation_malicious >= threshold))
        self.assertTrue(np.all(prior_regression_malicious >= threshold))

    def test_operating_threshold_uses_validation_cap_when_it_is_lower(self) -> None:
        threshold = TRAINING.select_operating_block_threshold(
            np.asarray([0.4, 0.8]),
            np.asarray([0.6, 0.9]),
        )
        self.assertEqual(
            threshold,
            float(np.nextafter(np.float64(0.4), np.float64(-np.inf))),
        )

    def test_operating_threshold_rejects_invalid_panels(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty 1D array"):
            TRAINING.select_operating_block_threshold(
                np.asarray([]),
                np.asarray([0.5]),
            )
        with self.assertRaisesRegex(ValueError, "positive detector threshold"):
            TRAINING.select_operating_block_threshold(
                np.asarray([0.5]),
                np.asarray([0.0]),
            )

    def test_threshold_minimizes_false_positives_under_recall_gates(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1])
        scores = np.asarray([0.1, 0.6, 0.75, 0.8, 0.7])
        groups = {
            "all_malicious": np.asarray([False, False, False, True, True]),
            "image_only": np.asarray([False, False, False, True, False]),
            "text_led": np.asarray([False, False, False, False, True]),
        }
        threshold = TRAINING.select_block_threshold(labels, scores, groups)
        self.assertEqual(
            threshold,
            float(np.nextafter(np.float64(0.7), np.float64(-np.inf))),
        )
        self.assertTrue(np.all(scores[labels == 1] >= threshold))
        self.assertEqual(int(np.sum(scores[labels == 0] >= threshold)), 1)

    def test_threshold_rejects_incomplete_malicious_gates(self) -> None:
        labels = np.asarray([0, 1, 1])
        scores = np.asarray([0.1, 0.8, 0.7])
        with self.assertRaisesRegex(ValueError, "do not cover"):
            TRAINING.select_block_threshold(
                labels,
                scores,
                {"image_only": np.asarray([False, True, False])},
            )

    def test_training_weights_distinguish_attack_channels(self) -> None:
        rows = [
            {
                "sample_id": "ordinary",
                "label_id": "0",
                "split": "train",
                "strategy": "official_bordair_benign_text_rendered_in_image",
                "hard_negative": "0",
                "hard_negative_category": "",
            },
            {
                "sample_id": "paired",
                "label_id": "0",
                "split": "train",
                "strategy": "official_bordair_benign_text_rendered_in_image",
                "hard_negative": "0",
                "hard_negative_category": "",
            },
            {
                "sample_id": "hard",
                "label_id": "0",
                "split": "train",
                "strategy": "hard_negative",
                "hard_negative": "1",
                "hard_negative_category": "policy_legal",
            },
            {
                "sample_id": "image-malicious",
                "label_id": "1",
                "split": "train",
                "strategy": TRAINING.IMAGE_ONLY_STRATEGY,
                "hard_negative": "0",
                "hard_negative_category": "",
            },
            {
                "sample_id": "text-malicious",
                "label_id": "1",
                "split": "train",
                "strategy": TRAINING.TEXT_LED_STRATEGY,
                "paired_benign_sample_id": "paired",
                "hard_negative": "0",
                "hard_negative_category": "",
            },
        ]
        labels = np.asarray([0, 0, 0, 1, 1])
        values = TRAINING.training_sample_weights(rows, labels)
        np.testing.assert_array_equal(
            values, np.asarray([1.0, 2.0, 2.5, 1.0, 2.0])
        )
        summary = TRAINING.sample_weighting_summary(
            rows, labels, values
        )
        self.assertEqual(summary["groups"]["ordinary_benign"]["rows"], 1)
        self.assertEqual(
            summary["groups"]["counterfactual_paired_benign"]["weighted_mass"],
            2.0,
        )
        self.assertEqual(
            summary["groups"]["hard_negative_benign"]["weighted_mass"], 2.5
        )
        self.assertEqual(summary["counterfactual_pairing"]["pairs"], 1)
        self.assertEqual(summary["benign_weighted_mass"], 5.5)
        self.assertEqual(summary["malicious_weighted_mass"], 3.0)

    def test_hard_negative_category_does_not_change_configured_weight(self) -> None:
        rows = [
            {
                "sample_id": category,
                "label_id": "0",
                "strategy": "hard_negative",
                "hard_negative": "1",
                "hard_negative_category": category,
            }
            for category in ("policy_legal", "code_errors", "prompt_meta")
        ]
        values = TRAINING.training_sample_weights(rows, np.zeros(3, dtype=int))
        np.testing.assert_array_equal(values, np.asarray([2.5, 2.5, 2.5]))

    def test_hard_negative_markers_are_strict_and_malicious_rows_cannot_use_them(
        self,
    ) -> None:
        base = {
            "sample_id": "sample",
            "label_id": "0",
            "strategy": "benign",
            "hard_negative": "0",
            "hard_negative_category": "",
        }
        invalid = [
            dict(base, hard_negative="true"),
            dict(base, hard_negative_category="policy_legal"),
            dict(
                base,
                hard_negative="1",
                hard_negative_category="unknown_category",
            ),
            dict(
                base,
                label_id="1",
                hard_negative="1",
                hard_negative_category="policy_legal",
            ),
        ]
        for row in invalid:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    TRAINING.validate_hard_negative_metadata([row])

    def test_counterfactual_pair_metadata_is_strict(self) -> None:
        benign = {
            "sample_id": "paired",
            "label_id": "0",
            "split": "train",
            "strategy": "benign",
            "hard_negative": "0",
            "hard_negative_category": "",
        }
        text_led = {
            "sample_id": "text-led",
            "label_id": "1",
            "split": "train",
            "strategy": TRAINING.TEXT_LED_STRATEGY,
            "paired_benign_sample_id": "paired",
            "hard_negative": "0",
            "hard_negative_category": "",
        }
        invalid_cases = {
            "lacks paired_benign_sample_id": [
                benign,
                dict(text_led, paired_benign_sample_id=""),
            ],
            "does not exist": [
                benign,
                dict(text_led, paired_benign_sample_id="missing"),
            ],
            "not benign": [
                dict(benign, label_id="1", strategy=TRAINING.IMAGE_ONLY_STRATEGY),
                text_led,
            ],
            "missing split metadata": [
                {key: value for key, value in benign.items() if key != "split"},
                text_led,
            ],
            "crosses splits": [
                dict(benign, split="validation"),
                text_led,
            ],
            "cannot be a hard negative": [
                dict(
                    benign,
                    hard_negative="1",
                    hard_negative_category="policy_legal",
                ),
                text_led,
            ],
            "referenced more than once": [
                benign,
                text_led,
                dict(text_led, sample_id="text-led-2"),
            ],
        }
        for expected, rows in invalid_cases.items():
            labels = np.asarray([int(row["label_id"]) for row in rows])
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    TRAINING.training_sample_weights(rows, labels)

    def test_counterfactual_paired_weight_must_be_positive(self) -> None:
        rows = [
            {
                "sample_id": "paired",
                "label_id": "0",
                "split": "train",
                "strategy": "benign",
                "hard_negative": "0",
                "hard_negative_category": "",
            },
            {
                "sample_id": "text-led",
                "label_id": "1",
                "split": "train",
                "strategy": TRAINING.TEXT_LED_STRATEGY,
                "paired_benign_sample_id": "paired",
                "hard_negative": "0",
                "hard_negative_category": "",
            },
        ]
        for invalid in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    TRAINING.training_sample_weights(
                        rows,
                        np.asarray([0, 1]),
                        counterfactual_paired_benign_weight=invalid,
                    )

    def test_cli_weight_defaults_and_override_parse_exactly(self) -> None:
        with patch.object(sys, "argv", ["train_detector.py", "llava05b"]):
            defaults = TRAINING.parse_args()
        self.assertEqual(defaults.image_only_malicious_weight, 1.0)
        self.assertEqual(defaults.text_led_malicious_weight, 2.0)
        self.assertEqual(defaults.counterfactual_paired_benign_weight, 2.0)
        self.assertEqual(defaults.benign_weight, 1.0)
        self.assertEqual(defaults.hard_negative_weight, 2.5)
        self.assertIsNone(defaults.candidate_image_only_malicious_weights)
        self.assertIsNone(defaults.candidate_hard_negative_weights)
        self.assertIsNone(defaults.candidate_learning_rates)
        self.assertIsNone(defaults.candidate_l2_values)
        self.assertEqual(
            tuple(defaults.candidate_poolings), TRAINING.CANDIDATE_POOLINGS
        )
        with patch.object(
            sys,
            "argv",
            [
                "train_detector.py",
                "llava05b",
                "--hard-negative-weight",
                "3.25",
                "--counterfactual-paired-benign-weight",
                "2.75",
                "--candidate-poolings",
                "text_tokens",
                "text_image_tokens",
                "--candidate-image-only-malicious-weights",
                "1.2",
                "1.4",
                "--candidate-hard-negative-weights",
                "3.5",
                "4.5",
                "--candidate-learning-rates",
                "0.02",
                "--candidate-l2-values",
                "0.001",
                "0.01",
            ],
        ):
            overridden = TRAINING.parse_args()
        self.assertEqual(overridden.hard_negative_weight, 3.25)
        self.assertEqual(overridden.counterfactual_paired_benign_weight, 2.75)
        self.assertEqual(
            overridden.candidate_poolings,
            ["text_tokens", "text_image_tokens"],
        )
        self.assertEqual(
            overridden.candidate_image_only_malicious_weights,
            [1.2, 1.4],
        )
        self.assertEqual(overridden.candidate_hard_negative_weights, [3.5, 4.5])
        self.assertEqual(overridden.candidate_learning_rates, [0.02])
        self.assertEqual(overridden.candidate_l2_values, [0.001, 0.01])

    def test_positive_candidate_grid_validation_is_strict(self) -> None:
        self.assertEqual(
            TRAINING.validate_positive_candidate_grid(
                [1, np.float64(2.5)],
                name="weights",
            ),
            (1.0, 2.5),
        )
        invalid_cases = (
            ([], "at least one"),
            ([1.0, 1], "duplicates"),
            ([0.0], "finite and positive"),
            ([-1.0], "finite and positive"),
            ([np.nan], "finite and positive"),
            ([np.inf], "finite and positive"),
            ([True], "numeric"),
            (["1.0"], "numeric"),
            ("1.0", "sequence"),
        )
        for values, expected in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, expected):
                    TRAINING.validate_positive_candidate_grid(
                        values,
                        name="weights",
                    )

    def test_candidate_grid_defaults_preserve_single_weight_behavior(self) -> None:
        with patch.object(sys, "argv", ["train_detector.py", "llava05b"]):
            args = TRAINING.parse_args()
        grids = TRAINING.resolved_candidate_grids(args)
        self.assertEqual(grids["image_only_malicious_weights"], (1.0,))
        self.assertEqual(grids["hard_negative_weights"], (2.5,))
        self.assertEqual(grids["learning_rates"], TRAINING.CANDIDATE_LEARNING_RATES)
        self.assertEqual(grids["l2_values"], TRAINING.CANDIDATE_L2_VALUES)

    def test_candidate_pooling_validation_requires_fused_for_promotion(self) -> None:
        self.assertEqual(
            TRAINING.validate_candidate_poolings(
                ["image_tokens"], require_fused=False
            ),
            ("image_tokens",),
        )
        invalid_cases = (
            ([], "at least one"),
            (["text_tokens", "text_tokens"], "duplicates"),
            (["unknown", "text_image_tokens"], "unsupported"),
            (["text_tokens", "image_tokens"], "must include text_image_tokens"),
        )
        for poolings, expected in invalid_cases:
            with self.subTest(poolings=poolings):
                with self.assertRaisesRegex(ValueError, expected):
                    TRAINING.validate_candidate_poolings(
                        poolings,
                        require_fused=True,
                    )

    def test_cli_allows_non_fused_pooling_only_for_extract_only(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "train_detector.py",
                "llava05b",
                "--extract-only",
                "--candidate-poolings",
                "image_tokens",
            ],
        ):
            args = TRAINING.parse_args()
        TRAINING.resolved_target_info(args)
        args.extract_only = False
        with self.assertRaisesRegex(ValueError, "must include text_image_tokens"):
            TRAINING.resolved_target_info(args)

    def test_train_rejects_programmatic_pooling_set_without_fused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must include text_image_tokens"):
            TRAINING.train(
                "llava05b",
                [],
                [],
                {},
                {},
                candidate_poolings=("text_tokens", "image_tokens"),
            )

    def test_candidate_ranking_uses_validation_only(self) -> None:
        common = {
            "pooling": "text_image_tokens",
            "validation_all_malicious_recall": 1.0,
            "validation_image_only_recall": 1.0,
            "validation_text_led_recall": 1.0,
            "validation_fpr_target_met": True,
            "validation_f1": 0.95,
            "validation_auprc": 0.99,
            "validation_auroc": 0.99,
            "validation_separation": 0.1,
            "l2": 0.001,
        }
        worse_validation = {
            **common,
            "candidate_id": 0,
            "validation_false_positive_rate": 0.02,
            "test_false_positive_rate": 0.0,
            "regression_recall": 1.0,
            "frozen_false_positive_rate": 0.0,
        }
        better_validation = {
            **common,
            "candidate_id": 1,
            "validation_false_positive_rate": 0.01,
            "test_false_positive_rate": 1.0,
            "regression_recall": 0.0,
            "frozen_false_positive_rate": 1.0,
        }
        non_fused = {
            **better_validation,
            "candidate_id": 2,
            "pooling": "image_tokens",
            "validation_false_positive_rate": 0.0,
        }
        selected = TRAINING.select_validation_candidate(
            [worse_validation, better_validation, non_fused]
        )
        self.assertEqual(selected["candidate_id"], 1)

        worse_validation.update(
            test_false_positive_rate=-100.0,
            regression_recall=100.0,
            frozen_false_positive_rate=-100.0,
        )
        better_validation.update(
            test_false_positive_rate=100.0,
            regression_recall=-100.0,
            frozen_false_positive_rate=100.0,
        )
        selected_after_excluded_mutation = TRAINING.select_validation_candidate(
            [worse_validation, better_validation]
        )
        self.assertEqual(selected_after_excluded_mutation["candidate_id"], 1)

    def test_end_to_end_training_promotes_fused_v7_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            development_rows: list[dict[str, str]] = []
            text_features: list[list[float]] = []
            image_features: list[list[float]] = []
            for split in ("train", "validation", "test"):
                for suffix, label, strategy, text, image in (
                    ("B", 0, "benign", [-2.0, -2.0], [-2.0, -2.0]),
                    (
                        "I",
                        1,
                        TRAINING.IMAGE_ONLY_STRATEGY,
                        [-2.0, -2.0],
                        [2.0, 2.0],
                    ),
                    (
                        "T",
                        1,
                        TRAINING.TEXT_LED_STRATEGY,
                        [2.0, 2.0],
                        [-2.0, -2.0],
                    ),
                ):
                    development_rows.append(
                        {
                            "sample_id": f"{split}-{suffix}",
                            "label_id": str(label),
                            "split": split,
                            "group_id": f"group-{split}-{suffix}",
                            "attack_style": "synthetic",
                            "strategy": strategy,
                            "paired_benign_sample_id": (
                                f"{split}-B"
                                if strategy == TRAINING.TEXT_LED_STRATEGY
                                else ""
                            ),
                            "hard_negative": "0",
                            "hard_negative_category": "",
                        }
                    )
                    text_features.append(text)
                    image_features.append(image)
                if split == "train":
                    development_rows.append(
                        {
                            "sample_id": "train-H",
                            "label_id": "0",
                            "split": "train",
                            "group_id": "group-train-H",
                            "attack_style": "synthetic",
                            "strategy": "hard_negative",
                            "paired_benign_sample_id": "",
                            "hard_negative": "1",
                            "hard_negative_category": "policy_legal",
                        }
                    )
                    text_features.append([-1.0, -1.0])
                    image_features.append([-1.0, -1.0])
            regression_rows = [
                {
                    "sample_id": "regression-I",
                    "label_id": "1",
                    "split": "regression",
                    "group_id": "regression-family",
                    "attack_style": "synthetic",
                    "strategy": TRAINING.IMAGE_ONLY_STRATEGY,
                    "hard_negative": "0",
                    "hard_negative_category": "",
                }
            ]

            provenance = {
                "model_family": "llava_onevision",
                "model_id": "synthetic-model",
                "model_revision": "synthetic-revision",
                "tokenizer_revision": "synthetic-revision",
                "preprocessing_sha256": TRAINING.expected_preprocessing_sha256(
                    "llava05b",
                    {
                        "model_id": "synthetic-model",
                        "model_revision": "synthetic-revision",
                        "tokenizer_revision": "synthetic-revision",
                    },
                ),
                "layer": -1,
            }

            def save_features(
                name: str,
                sample_ids: list[str],
                text: np.ndarray,
                image: np.ndarray,
            ) -> dict[str, Path]:
                paths: dict[str, Path] = {}
                for pooling, values in (
                    ("text_tokens", text),
                    ("image_tokens", image),
                ):
                    path = run / f"{name}_{pooling}.npz"
                    np.savez(
                        path,
                        embeddings=values,
                        sample_id=np.asarray(sample_ids),
                        **{
                            key: np.asarray([value])
                            for key, value in provenance.items()
                        },
                        pooling=np.asarray([pooling]),
                    )
                    paths[pooling] = path
                return paths

            development_paths = save_features(
                "development",
                [row["sample_id"] for row in development_rows],
                np.asarray(text_features),
                np.asarray(image_features),
            )
            regression_paths = save_features(
                "regression",
                ["regression-I"],
                np.asarray([[-2.0, -2.0]]),
                np.asarray([[2.0, 2.0]]),
            )
            manifest = run / "corpus_manifest_v7.json"
            development_metadata = run / "development_metadata_v7.csv"
            regression_metadata = run / "regression_metadata_v7.csv"
            manifest.write_text("{}\n", encoding="utf-8")
            development_metadata.write_text("synthetic\n", encoding="utf-8")
            regression_metadata.write_text("synthetic\n", encoding="utf-8")
            observed_fit_weights: list[np.ndarray] = []
            original_fit = TRAINING.LogisticRegressionNumpy.fit

            def recording_fit(
                classifier: LogisticRegressionNumpy,
                features: np.ndarray,
                labels: np.ndarray,
                sample_weight: np.ndarray | None = None,
            ) -> LogisticRegressionNumpy:
                if sample_weight is None:
                    raise AssertionError("candidate fit omitted sample weights")
                observed_fit_weights.append(np.asarray(sample_weight).copy())
                return original_fit(
                    classifier,
                    features,
                    labels,
                    sample_weight=sample_weight,
                )

            with (
                patch.object(TRAINING, "RUN", run),
                patch.object(TRAINING, "CORPUS_MANIFEST", manifest),
                patch.object(
                    TRAINING, "DEVELOPMENT_METADATA", development_metadata
                ),
                patch.object(TRAINING, "REGRESSION_METADATA", regression_metadata),
                patch.object(
                    TRAINING.LogisticRegressionNumpy,
                    "fit",
                    new=recording_fit,
                ),
            ):
                with redirect_stdout(io.StringIO()):
                    summary = TRAINING.train(
                        "llava05b",
                        development_rows,
                        regression_rows,
                        development_paths,
                        regression_paths,
                        candidate_poolings=("text_image_tokens",),
                        candidate_image_only_malicious_weights=(1.0, 2.0),
                        candidate_hard_negative_weights=(2.5, 4.5),
                        candidate_learning_rates=(0.02,),
                        candidate_l2_values=(0.001,),
                    )
            self.assertTrue(summary["acceptance"]["passed"])
            self.assertEqual(len(observed_fit_weights), 4)
            np.testing.assert_array_equal(
                observed_fit_weights[0],
                np.asarray([2.0, 1.0, 2.0, 2.5]),
            )
            np.testing.assert_array_equal(
                observed_fit_weights[1],
                np.asarray([2.0, 1.0, 2.0, 4.5]),
            )
            np.testing.assert_array_equal(
                observed_fit_weights[2],
                np.asarray([2.0, 2.0, 2.0, 2.5]),
            )
            np.testing.assert_array_equal(
                observed_fit_weights[3],
                np.asarray([2.0, 2.0, 2.0, 4.5]),
            )
            with (
                run / "training_v7/llava05b/candidate_metrics.csv"
            ).open(encoding="utf-8", newline="") as handle:
                candidate_rows = list(csv.DictReader(handle))
            self.assertEqual(len(candidate_rows), 4)
            self.assertEqual(
                {
                    (
                        float(row["image_only_malicious_weight"]),
                        float(row["hard_negative_weight"]),
                    )
                    for row in candidate_rows
                },
                {(1.0, 2.5), (1.0, 4.5), (2.0, 2.5), (2.0, 4.5)},
            )
            self.assertEqual(
                summary["selected_candidate"]["pooling"], "text_image_tokens"
            )
            self.assertEqual(summary["evaluated_poolings"], ["text_image_tokens"])
            self.assertIn(
                "configured representations (text_image_tokens)",
                summary["selection_rule"],
            )
            self.assertIn(
                "greatest representable float strictly below",
                summary["selection_rule"],
            )
            self.assertIn(
                "acts only as an acceptance and selection-qualification gate",
                summary["selection_rule"],
            )
            self.assertIn(
                "post-selection operating point",
                summary["selection_rule"],
            )
            selected = summary["selected_candidate"]
            expected_identity = TRAINING.training_identity_sha256(
                summary["training_configuration"],
                selected,
            )
            self.assertEqual(
                summary["training_identity"],
                {"schema_version": 1, "sha256": expected_identity},
            )
            self.assertTrue(
                summary["artifact_source"].endswith(
                    f"cfgcand-{expected_identity}"
                )
            )
            self.assertEqual(
                summary["training_configuration"]["sample_weight_grid"],
                {
                    "image_only_malicious": [1.0, 2.0],
                    "hard_negative_benign": [2.5, 4.5],
                    "fixed": {
                        "ordinary_benign": 1.0,
                        "counterfactual_paired_benign": 2.0,
                        "text_led_malicious": 2.0,
                    },
                },
            )
            self.assertEqual(
                summary["training_configuration"]["sample_weights"][
                    "image_only_malicious"
                ],
                selected["image_only_malicious_weight"],
            )
            self.assertEqual(
                summary["training_configuration"]["sample_weights"][
                    "hard_negative_benign"
                ],
                selected["hard_negative_weight"],
            )
            mutated_configuration = json.loads(
                json.dumps(summary["training_configuration"])
            )
            mutated_configuration["sample_weight_grid"][
                "image_only_malicious"
            ][0] = 9.0
            self.assertNotEqual(
                TRAINING.training_identity_sha256(
                    mutated_configuration,
                    selected,
                ),
                expected_identity,
            )
            self.assertEqual(
                selected["validation_threshold_rule"],
                "maximum_specificity_weakest_malicious_predecessor",
            )
            self.assertTrue(
                selected[
                    "validation_threshold_is_weakest_malicious_predecessor"
                ]
            )
            self.assertGreater(selected["validation_threshold_ulp_gap"], 0.0)
            self.assertEqual(
                selected["validation_unavoidable_false_positives"],
                selected["validation_false_positives"],
            )
            operating_point = summary["operating_point"]
            self.assertEqual(
                operating_point["rule"],
                "predecessor_of_lower_validation_and_prior_regression_malicious",
            )
            self.assertTrue(operating_point["threshold_is_cap_predecessor"])
            self.assertFalse(
                operating_point["candidate_ranking_uses_prior_regression"]
            )
            self.assertFalse(
                operating_point["candidate_ranking_uses_internal_test"]
            )
            self.assertFalse(
                operating_point["candidate_ranking_uses_frozen_panels"]
            )
            self.assertEqual(
                selected["block_threshold"],
                operating_point["block_threshold"],
            )
            self.assertEqual(
                selected["validation_selection_block_threshold"],
                selected["validation_block_threshold"],
            )
            self.assertEqual(summary["feature_dimension"], 4)
            self.assertEqual(summary["test_metrics"]["false_negatives"], 0.0)
            self.assertEqual(
                summary["sample_weighting"]["configured_weights"][
                    "counterfactual_paired_benign"
                ],
                2.0,
            )
            self.assertEqual(
                summary["sample_weighting"]["counterfactual_pairing"]["pairs"],
                1,
            )
            output_manifest_path = (
                run / "training_v7/llava05b/output_manifest.json"
            )
            self.assertTrue(output_manifest_path.is_file())
            output_manifest = json.loads(
                output_manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                output_manifest["training_identity_sha256"],
                expected_identity,
            )


class QwenChunkResumeTests(unittest.TestCase):
    def test_corrupt_completed_chunk_is_reextracted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_root = Path(directory) / "features"
            raw = feature_root / "raw"
            rows = [
                {
                    "sample_id": "qwen-row-1",
                    "prompt_text": "Inspect this image.",
                    "image_path": "images/example.png",
                    "image_sha256": "b" * 64,
                }
            ]
            target_info = {
                "model_family": "qwen25_vl",
                "model_id": "synthetic-qwen",
                "model_revision": "synthetic-revision",
                "tokenizer_revision": "synthetic-revision",
                "feature_dim": 2,
            }
            chunk_metadata = (
                feature_root / "chunk_metadata" / "chunk_0001.csv"
            )
            TRAINING.write_adapter_metadata(chunk_metadata, rows)
            output_dir = (
                feature_root
                / "chunks"
                / f"0001-{TRAINING.sha256(chunk_metadata)[:16]}"
            )
            expected = {
                pooling: output_dir
                / f"chunk_0001_layer_m1_{pooling}.npz"
                for pooling in TRAINING.EXTRACTION_POOLINGS
            }

            def write_valid_embedding(path: Path, pooling: str) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    path,
                    embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
                    sample_id=np.asarray(["qwen-row-1"]),
                    model_family=np.asarray(["qwen25_vl"]),
                    model_id=np.asarray(["synthetic-qwen"]),
                    model_revision=np.asarray(["synthetic-revision"]),
                    tokenizer_revision=np.asarray(["synthetic-revision"]),
                    preprocessing_sha256=np.asarray(
                        [
                            TRAINING.expected_preprocessing_sha256(
                                "qwen25vl3b", target_info
                            )
                        ]
                    ),
                    layer=np.asarray([-1]),
                    pooling=np.asarray([pooling]),
                )

            write_valid_embedding(expected["text_tokens"], "text_tokens")
            expected["image_tokens"].write_bytes(b"interrupted-npz")

            def extract_replacement(**kwargs: object) -> dict[str, Path]:
                replacement: dict[str, Path] = {}
                replacement_root = Path(str(kwargs["output_dir"]))
                output_prefix = str(kwargs["output_prefix"])
                for pooling in kwargs["poolings"]:  # type: ignore[index]
                    pooling_name = str(pooling)
                    path = (
                        replacement_root
                        / f"{output_prefix}_layer_m1_{pooling_name}.npz"
                    )
                    write_valid_embedding(path, pooling_name)
                    replacement[pooling_name] = path
                return replacement

            output = io.StringIO()
            with (
                patch.object(
                    TRAINING,
                    "extract_qwen25_vl_pooling_embeddings",
                    side_effect=extract_replacement,
                ) as delegate,
                redirect_stdout(output),
            ):
                combined = TRAINING.extract_qwen_panel_resumably(
                    target="qwen25vl3b",
                    panel="development_new",
                    feature_root=feature_root,
                    raw=raw,
                    rows=rows,
                    config=TRAINING.Qwen25VLExtractionConfig(
                        model_id="synthetic-qwen",
                        model_revision="synthetic-revision",
                        tokenizer_revision="synthetic-revision",
                    ),
                    target_info=target_info,
                    chunk_size=32,
                )

            delegate.assert_called_once()
            self.assertIn(
                "discarding invalid qwen25vl3b development_new chunk 1/1",
                output.getvalue(),
            )
            self.assertIn(
                "completed qwen25vl3b development_new chunk 1/1",
                output.getvalue(),
            )
            for pooling, path in combined.items():
                TRAINING.validate_embedding_file(
                    path,
                    rows,
                    "qwen25vl3b",
                    pooling,
                    target_info,
                )


class V6FeatureReuseTests(unittest.TestCase):
    def test_reuse_requires_exact_manifest_features_metadata_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            image = run / "images" / "old.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"verified-image")
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            metadata = run / "development_metadata.csv"
            old_row = {
                "sample_id": "BTI-00001",
                "label_id": "0",
                "split": "train",
                "group_id": "benign-1",
                "attack_style": "benign_control",
                "strategy": "official_bordair_benign_text_rendered_in_image",
                "source": "test",
                "prompt_text": "Inspect this image.",
                "image_path": "images/old.png",
                "image_text": "harmless",
                "render_style": "0",
                "image_sha256": image_hash,
            }
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(old_row))
                writer.writeheader()
                writer.writerow(old_row)

            target_info = {
                "model_family": "llava_onevision",
                "model_id": "test-model",
                "model_revision": "test-revision",
                "tokenizer_revision": "test-revision",
                "feature_dim": 2,
            }
            preprocessing_sha256 = TRAINING.expected_preprocessing_sha256(
                "llava05b", target_info
            )
            feature_panel = (
                run / "features_v6" / "llava05b" / "development"
            )
            feature_root = feature_panel / "raw"
            feature_root.mkdir(parents=True)
            feature_paths = {
                pooling: feature_root
                / f"development_layer_m1_{pooling}.npz"
                for pooling in TRAINING.EXTRACTION_POOLINGS
            }

            def write_features(fingerprint: str) -> None:
                for pooling, path in feature_paths.items():
                    np.savez(
                        path,
                        embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
                        sample_id=np.asarray([old_row["sample_id"]]),
                        model_family=np.asarray([target_info["model_family"]]),
                        model_id=np.asarray([target_info["model_id"]]),
                        model_revision=np.asarray(
                            [target_info["model_revision"]]
                        ),
                        tokenizer_revision=np.asarray(
                            [target_info["tokenizer_revision"]]
                        ),
                        preprocessing_sha256=np.asarray([fingerprint]),
                        layer=np.asarray([-1]),
                        pooling=np.asarray([pooling]),
                    )

            write_features(preprocessing_sha256)
            adapter_metadata = feature_panel / "adapter_metadata.csv"
            TRAINING.write_adapter_metadata(adapter_metadata, [old_row])
            feature_manifest = feature_panel / "feature_manifest.json"

            def write_manifest() -> str:
                feature_manifest.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "target": "llava05b",
                            "panel": "development",
                            "rows": 1,
                            "source_metadata": "development_metadata.csv",
                            "source_metadata_sha256": TRAINING.sha256(metadata),
                            "adapter_metadata_sha256": TRAINING.sha256(
                                adapter_metadata
                            ),
                            "embeddings": {
                                pooling: {
                                    "path": str(path.relative_to(run)).replace(
                                        "\\", "/"
                                    ),
                                    "sha256": TRAINING.sha256(path),
                                }
                                for pooling, path in feature_paths.items()
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return TRAINING.sha256(feature_manifest)

            pinned_manifest_hash = write_manifest()

            destination = run / "images_v7" / "old.png"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(image.read_bytes())
            matching = dict(old_row, image_path="images_v7/old.png")
            changed_text = dict(matching, prompt_text="Changed caller text")
            changed_hash = dict(matching, image_sha256="b" * 64)
            with (
                patch.object(TRAINING, "RUN", run),
                patch.object(
                    TRAINING,
                    "V6_METADATA",
                    {"development": metadata},
                ),
                patch.object(
                    TRAINING,
                    "V6_FEATURE_MANIFEST_SHA256",
                    {
                        "llava05b": {
                            "development": pinned_manifest_hash,
                        }
                    },
                ),
            ):
                reused, _, _ = TRAINING._validated_v6_reuse(
                    target="llava05b",
                    panel="development",
                    rows=[matching, changed_text, changed_hash],
                    target_info=target_info,
                )
            self.assertEqual(reused, {0: 0})

            adapter_metadata.write_text("tampered\n", encoding="utf-8")
            with (
                patch.object(TRAINING, "RUN", run),
                patch.object(
                    TRAINING,
                    "V6_METADATA",
                    {"development": metadata},
                ),
                patch.object(
                    TRAINING,
                    "V6_FEATURE_MANIFEST_SHA256",
                    {
                        "llava05b": {
                            "development": pinned_manifest_hash,
                        }
                    },
                ),
                redirect_stdout(io.StringIO()),
            ):
                adapter_tamper, _, _ = TRAINING._validated_v6_reuse(
                    target="llava05b",
                    panel="development",
                    rows=[matching],
                    target_info=target_info,
                )
            self.assertEqual(adapter_tamper, {})

            TRAINING.write_adapter_metadata(adapter_metadata, [old_row])
            feature_paths["image_tokens"].write_bytes(b"tampered-npz")
            with (
                patch.object(TRAINING, "RUN", run),
                patch.object(
                    TRAINING,
                    "V6_METADATA",
                    {"development": metadata},
                ),
                patch.object(
                    TRAINING,
                    "V6_FEATURE_MANIFEST_SHA256",
                    {
                        "llava05b": {
                            "development": pinned_manifest_hash,
                        }
                    },
                ),
                redirect_stdout(io.StringIO()),
            ):
                feature_tamper, _, _ = TRAINING._validated_v6_reuse(
                    target="llava05b",
                    panel="development",
                    rows=[matching],
                    target_info=target_info,
                )
            self.assertEqual(feature_tamper, {})

            write_features("0" * 64)
            wrong_fingerprint_manifest_hash = write_manifest()
            with (
                patch.object(TRAINING, "RUN", run),
                patch.object(
                    TRAINING,
                    "V6_METADATA",
                    {"development": metadata},
                ),
                patch.object(
                    TRAINING,
                    "V6_FEATURE_MANIFEST_SHA256",
                    {
                        "llava05b": {
                            "development": wrong_fingerprint_manifest_hash,
                        }
                    },
                ),
                redirect_stdout(io.StringIO()),
            ):
                wrong_fingerprint, _, _ = TRAINING._validated_v6_reuse(
                    target="llava05b",
                    panel="development",
                    rows=[matching],
                    target_info=target_info,
                )
            self.assertEqual(wrong_fingerprint, {})


class ChainedFeatureReuseTests(unittest.TestCase):
    def test_schema3_reuse_requires_exact_rows_and_both_image_copies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            archive = run / "schema3-archive"
            feature_root = archive / "llava05b" / "development"
            feature_root.mkdir(parents=True)
            image_relative = Path("images_v7/development/BTI-00001.png")
            payload = b"same-rendered-image"
            current_image = run / image_relative
            archived_image = archive / image_relative
            current_image.parent.mkdir(parents=True)
            archived_image.parent.mkdir(parents=True)
            current_image.write_bytes(payload)
            archived_image.write_bytes(payload)
            image_hash = hashlib.sha256(payload).hexdigest()
            source_row = {
                "sample_id": "BTI-00001",
                "label_id": "0",
                "split": "train",
                "group_id": "benign-1",
                "attack_style": "benign_control",
                "strategy": "official_bordair_benign_text_rendered_in_image",
                "source": "official benign",
                "prompt_text": "Inspect this image.",
                "image_path": image_relative.as_posix(),
                "image_text": "Harmless account guidance.",
                "render_style": "0",
                "image_sha256": image_hash,
                "hard_negative": "1",
                "hard_negative_category": "credentials_auth",
            }
            source_metadata = archive / "development_metadata_v7.csv"
            with source_metadata.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(source_row))
                writer.writeheader()
                writer.writerow(source_row)
            source_metadata_hash = hashlib.sha256(
                source_metadata.read_bytes()
            ).hexdigest()
            corpus_manifest = archive / "corpus_manifest_v7.json"
            corpus_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "development": {
                            "metadata": source_metadata.name,
                            "metadata_sha256": source_metadata_hash,
                            "rows": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            adapter_metadata = feature_root / "adapter_metadata.csv"
            TRAINING.write_adapter_metadata(adapter_metadata, [source_row])
            target_info = {
                "model_family": "llava_onevision",
                "model_id": "test-model",
                "model_revision": "test-revision",
                "tokenizer_revision": "test-revision",
                "feature_dim": 2,
            }
            preprocessing_sha256 = TRAINING.expected_preprocessing_sha256(
                "llava05b", target_info
            )
            raw = feature_root / "raw"
            raw.mkdir()
            feature_paths: dict[str, Path] = {}
            for pooling in TRAINING.EXTRACTION_POOLINGS:
                path = raw / f"development_layer_m1_{pooling}.npz"
                np.savez(
                    path,
                    embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
                    sample_id=np.asarray([source_row["sample_id"]]),
                    model_family=np.asarray([target_info["model_family"]]),
                    model_id=np.asarray([target_info["model_id"]]),
                    model_revision=np.asarray([target_info["model_revision"]]),
                    tokenizer_revision=np.asarray(
                        [target_info["tokenizer_revision"]]
                    ),
                    preprocessing_sha256=np.asarray([preprocessing_sha256]),
                    layer=np.asarray([-1]),
                    pooling=np.asarray([pooling]),
                )
                feature_paths[pooling] = path
            (feature_root / "feature_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "target": "llava05b",
                        "panel": "development",
                        "rows": 1,
                        "source_metadata": source_metadata.name,
                        "source_metadata_sha256": source_metadata_hash,
                        "adapter_metadata_sha256": hashlib.sha256(
                            adapter_metadata.read_bytes()
                        ).hexdigest(),
                        "feature_reuse": {
                            "rows_reused_total": 0,
                            "rows_extracted": 1,
                        },
                        "embeddings": {
                            pooling: {
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                            }
                            for pooling, path in feature_paths.items()
                        },
                    }
                ),
                encoding="utf-8",
            )
            archive_manifest = archive / "archive_manifest.json"
            archive_manifest.write_text(
                json.dumps(
                    {
                        "files": {
                            source_metadata.name: {
                                "sha256": source_metadata_hash,
                            },
                            corpus_manifest.name: {
                                "sha256": hashlib.sha256(
                                    corpus_manifest.read_bytes()
                                ).hexdigest(),
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            archive_manifest_hash = hashlib.sha256(
                archive_manifest.read_bytes()
            ).hexdigest()
            changed = dict(source_row, source="changed")
            with (
                patch.object(TRAINING, "RUN", run),
                patch.object(TRAINING, "SCHEMA3_FEATURE_ROOT", archive),
                patch.object(
                    TRAINING,
                    "SCHEMA3_ARCHIVE_MANIFEST_SHA256",
                    archive_manifest_hash,
                ),
            ):
                reused, _, _, details = TRAINING._validated_schema3_reuse(
                    target="llava05b",
                    panel="development",
                    rows=[source_row, changed],
                    target_info=target_info,
                )
            self.assertEqual(reused, {0: 0})
            self.assertEqual(details["eligible_source_rows"], 1)

            text_path = feature_paths["text_tokens"]
            np.savez(
                text_path,
                embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
                sample_id=np.asarray([source_row["sample_id"]]),
                model_family=np.asarray([target_info["model_family"]]),
                model_id=np.asarray([target_info["model_id"]]),
                model_revision=np.asarray([target_info["model_revision"]]),
                tokenizer_revision=np.asarray(
                    [target_info["tokenizer_revision"]]
                ),
                preprocessing_sha256=np.asarray(["0" * 64]),
                layer=np.asarray([-1]),
                pooling=np.asarray(["text_tokens"]),
            )
            tampered_manifest = json.loads(
                (feature_root / "feature_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            tampered_manifest["embeddings"]["text_tokens"][
                "sha256"
            ] = TRAINING.sha256(text_path)
            (feature_root / "feature_manifest.json").write_text(
                json.dumps(tampered_manifest), encoding="utf-8"
            )
            with (
                patch.object(TRAINING, "RUN", run),
                patch.object(TRAINING, "SCHEMA3_FEATURE_ROOT", archive),
                patch.object(
                    TRAINING,
                    "SCHEMA3_ARCHIVE_MANIFEST_SHA256",
                    archive_manifest_hash,
                ),
                redirect_stdout(io.StringIO()),
            ):
                wrong_fingerprint, _, _, _ = (
                    TRAINING._validated_schema3_reuse(
                        target="llava05b",
                        panel="development",
                        rows=[source_row],
                        target_info=target_info,
                    )
                )
            self.assertEqual(wrong_fingerprint, {})

    def test_prior_v7_reuse_requires_archived_train_metadata_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            archive = run / "features_v7_trials" / "pre_hard_negative"
            feature_root = archive / "llava05b" / "development"
            feature_root.mkdir(parents=True)
            images = run / "images_v7" / "development"
            images.mkdir(parents=True)

            source_rows: list[dict[str, str]] = []
            for sample_id, split, payload in (
                ("MTI-00001", "train", b"train-image"),
                ("MTI-00002", "validation", b"validation-image"),
            ):
                image_path = images / f"{sample_id}.png"
                image_path.write_bytes(payload)
                source_rows.append(
                    {
                        "sample_id": sample_id,
                        "label_id": "1",
                        "split": split,
                        "group_id": f"family-{sample_id}",
                        "attack_style": "synthetic",
                        "strategy": TRAINING.TEXT_LED_STRATEGY,
                        "source": "archived-source",
                        "prompt_text": f"caller-{sample_id}",
                        "image_path": str(image_path.relative_to(run)).replace(
                            "\\", "/"
                        ),
                        "image_text": "benign image",
                        "render_style": "0",
                        "image_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )

            source_metadata = archive / "development_metadata_v7.csv"
            source_metadata.parent.mkdir(parents=True, exist_ok=True)
            with source_metadata.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
                writer.writeheader()
                writer.writerows(source_rows)
            source_hash = hashlib.sha256(source_metadata.read_bytes()).hexdigest()
            (archive / "corpus_manifest_v7.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "development": {
                            "metadata": "development_metadata_v7.csv",
                            "metadata_sha256": source_hash,
                            "rows": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            adapter_metadata = feature_root / "adapter_metadata.csv"
            TRAINING.write_adapter_metadata(adapter_metadata, source_rows)

            target_info = {
                "model_family": "llava_onevision",
                "model_id": "test-model",
                "model_revision": "test-revision",
                "tokenizer_revision": "test-revision",
                "feature_dim": 2,
            }
            preprocessing_sha256 = TRAINING.expected_preprocessing_sha256(
                "llava05b", target_info
            )
            raw = feature_root / "raw"
            raw.mkdir()
            feature_paths: dict[str, Path] = {}
            for pooling in TRAINING.EXTRACTION_POOLINGS:
                path = raw / f"development_layer_m1_{pooling}.npz"
                np.savez(
                    path,
                    embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                    sample_id=np.asarray(
                        [row["sample_id"] for row in source_rows]
                    ),
                    model_family=np.asarray([target_info["model_family"]]),
                    model_id=np.asarray([target_info["model_id"]]),
                    model_revision=np.asarray([target_info["model_revision"]]),
                    tokenizer_revision=np.asarray(
                        [target_info["tokenizer_revision"]]
                    ),
                    preprocessing_sha256=np.asarray([preprocessing_sha256]),
                    layer=np.asarray([-1]),
                    pooling=np.asarray([pooling]),
                )
                feature_paths[pooling] = path
            feature_manifest = {
                "schema_version": 1,
                "target": "llava05b",
                "panel": "development",
                "rows": 2,
                "source_metadata": "development_metadata_v7.csv",
                "source_metadata_sha256": source_hash,
                "adapter_metadata_sha256": hashlib.sha256(
                    adapter_metadata.read_bytes()
                ).hexdigest(),
                "v6_feature_reuse": {"rows_extracted": 2},
                "embeddings": {
                    pooling: {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                    }
                    for pooling, path in feature_paths.items()
                },
            }
            (feature_root / "feature_manifest.json").write_text(
                json.dumps(feature_manifest), encoding="utf-8"
            )

            matching_train = dict(source_rows[0])
            old_validation = dict(source_rows[1])
            changed_metadata = dict(source_rows[0], source="changed-source")
            with (
                patch.object(TRAINING, "RUN", run),
                patch.object(TRAINING, "PRIOR_V7_FEATURE_ROOT", archive),
            ):
                reused, _, _, details = TRAINING._validated_prior_v7_reuse(
                    target="llava05b",
                    panel="development",
                    rows=[matching_train, old_validation, changed_metadata],
                    target_info=target_info,
                )
                regression_reuse, _, _, _ = TRAINING._validated_prior_v7_reuse(
                    target="llava05b",
                    panel="regression",
                    rows=[matching_train],
                    target_info=target_info,
                )
            self.assertEqual(reused, {0: 0})
            self.assertEqual(regression_reuse, {})
            self.assertEqual(details["eligible_source_rows"], 1)

    def test_extract_panel_prefers_schema3_then_prior_then_v6(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            metadata_path = run / "development_metadata_v7.csv"
            metadata_path.write_text("metadata\n", encoding="utf-8")
            v6_metadata = run / "development_metadata.csv"
            v6_metadata.write_text("v6 metadata\n", encoding="utf-8")
            rows = [
                {
                    "sample_id": f"sample-{index}",
                    "label_id": "1",
                    "split": "train",
                    "group_id": f"group-{index}",
                    "attack_style": "synthetic",
                    "strategy": TRAINING.TEXT_LED_STRATEGY,
                    "prompt_text": f"text-{index}",
                    "image_path": f"image-{index}.png",
                    "image_sha256": str(index) * 64,
                }
                for index in range(4)
            ]
            target_info = {
                "model_family": "llava_onevision",
                "model_id": "test-model",
                "model_revision": "test-revision",
                "tokenizer_revision": "test-revision",
                "feature_dim": 1,
            }
            provenance = {
                "model_family": "llava_onevision",
                "model_id": "test-model",
                "model_revision": "test-revision",
                "tokenizer_revision": "test-revision",
                "preprocessing_sha256": TRAINING.expected_preprocessing_sha256(
                    "llava05b", target_info
                ),
                "layer": -1,
            }

            def feature_files(name: str, values: list[float]) -> dict[str, Path]:
                paths: dict[str, Path] = {}
                for pooling in TRAINING.EXTRACTION_POOLINGS:
                    path = run / f"{name}-{pooling}.npz"
                    np.savez(
                        path,
                        embeddings=np.asarray(values, dtype=np.float32).reshape(-1, 1),
                        sample_id=np.asarray(
                            [f"{name}-{index}" for index in range(len(values))]
                        ),
                        **{
                            key: np.asarray([value])
                            for key, value in provenance.items()
                        },
                        pooling=np.asarray([pooling]),
                    )
                    paths[pooling] = path
                return paths

            schema3_paths = feature_files("schema3", [10.0])
            prior_paths = feature_files("prior", [20.0])
            v6_paths = feature_files("v6", [30.0])
            new_paths = feature_files("new", [40.0])
            prior_details = {
                "source_metadata_sha256": "b" * 64,
                "source_feature_sha256": {
                    pooling: hashlib.sha256(path.read_bytes()).hexdigest()
                    for pooling, path in prior_paths.items()
                },
            }
            schema3_details = {
                "source_archive_manifest_sha256": "c" * 64,
                "source_feature_sha256": {
                    pooling: hashlib.sha256(path.read_bytes()).hexdigest()
                    for pooling, path in schema3_paths.items()
                },
            }
            with (
                patch.object(TRAINING, "RUN", run),
                patch.object(TRAINING, "V6_METADATA", {"development": v6_metadata}),
                patch.object(
                    TRAINING,
                    "_validated_schema3_reuse",
                    return_value=(
                        {0: 0},
                        schema3_paths,
                        [rows[0]],
                        schema3_details,
                    ),
                ),
                patch.object(
                    TRAINING,
                    "_validated_prior_v7_reuse",
                    return_value=(
                        {0: 0, 1: 0},
                        prior_paths,
                        [rows[0]],
                        prior_details,
                    ),
                ),
                patch.object(
                    TRAINING,
                    "_validated_v6_reuse",
                    return_value=({0: 0, 1: 0, 2: 0}, v6_paths, rows[:3]),
                ),
                patch.object(TRAINING, "_extract_rows", return_value=new_paths),
            ):
                with redirect_stdout(io.StringIO()):
                    output_paths = TRAINING.extract_panel(
                        "llava05b",
                        "development",
                        metadata_path,
                        rows,
                        target_info,
                        32,
                    )

            for path in output_paths.values():
                with np.load(path, allow_pickle=False) as data:
                    np.testing.assert_array_equal(
                        np.asarray(data["embeddings"]).reshape(-1),
                        np.asarray([10.0, 20.0, 30.0, 40.0]),
                    )
            manifest = json.loads(
                (
                    run
                    / "features_v7/llava05b/development/feature_manifest.json"
                ).read_text(encoding="utf-8")
            )
            reuse = manifest["feature_reuse"]
            self.assertEqual(reuse["rows_reused_total"], 3)
            self.assertEqual(reuse["rows_extracted"], 1)
            self.assertEqual(
                reuse["sources"]["schema3_pre_additional_hard_negatives"][
                    "rows_reused"
                ],
                1,
            )
            self.assertEqual(
                reuse["sources"]["prior_v7_pre_hard_negative"]["rows_reused"],
                1,
            )
            self.assertEqual(reuse["sources"]["v6"]["rows_reused"], 1)
            self.assertFalse(reuse["evaluation_panel_features_reused_for_training"])


if __name__ == "__main__":
    unittest.main()
