from __future__ import annotations

import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(PIPELINE_ROOT / "scripts"))

from aegis_research.metrics import average_precision, classification_metrics, roc_auc  # noqa: E402
from aegis_research.model import LogisticDetector, select_threshold  # noqa: E402
from aegis_research.signals import (  # noqa: E402
    POOLING_FEATURE_VIEWS,
    binary_entropy,
    build_feature_views,
    cross_modal_consistency,
)
import score_feature_bundle  # noqa: E402
import extract_mllm_features  # noqa: E402


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

    def test_fused_pooling_maps_to_concatenated_joint_representation(self) -> None:
        text = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        image = np.asarray([[5.0, 6.0], [7.0, 8.0]])
        attribution = np.zeros((2, 1))
        views = build_feature_views(text, image, attribution)
        fused_view = POOLING_FEATURE_VIEWS["text_image_tokens"]
        self.assertEqual(fused_view, "joint_representation")
        np.testing.assert_array_equal(
            views[fused_view], np.concatenate([text, image], axis=1)
        )


class DetectorTests(unittest.TestCase):
    def test_learns_separable_data(self) -> None:
        x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.asarray([0, 0, 1, 1])
        detector = LogisticDetector(epochs=500).fit(x, y)
        predictions = detector.predict_proba(x)
        self.assertTrue(np.all(predictions[:2] < 0.5))
        self.assertTrue(np.all(predictions[2:] > 0.5))


class ExplicitLegacySingleScoringTests(unittest.TestCase):
    def test_explicit_legacy_single_scores_deployed_fused_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "label_id"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"sample_id": "one", "label_id": "0"},
                        {"sample_id": "two", "label_id": "1"},
                    ]
                )
            provenance = {
                "model_family": np.asarray(["test_family"]),
                "model_id": np.asarray(["test/model"]),
                "model_revision": np.asarray(["revision"]),
                "tokenizer_revision": np.asarray(["tokenizer"]),
                "preprocessing_sha256": np.asarray(["a" * 64]),
                "layer": np.asarray([-1], dtype=np.int64),
                "pooling": np.asarray(["text_image_tokens"]),
            }
            features = root / "features.npz"
            np.savez(
                features,
                sample_ids=np.asarray(["one", "two"]),
                text_embeddings=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
                image_embeddings=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
                attribution_features=np.zeros((2, 1)),
                feature_source=np.asarray(["synthetic"]),
                id_column=np.asarray(["sample_id"]),
                **provenance,
            )
            detector = root / "detector.npz"
            np.savez(
                detector,
                weights=np.ones(4),
                mean=np.zeros(4),
                scale=np.ones(4),
                bias=np.asarray([0.0]),
                threshold=np.asarray([0.5]),
                **provenance,
            )
            output = root / "scores.csv"
            argv = [
                "score_feature_bundle.py",
                "--mode",
                "legacy_single",
                "--metadata",
                str(metadata),
                "--features",
                str(features),
                "--detector",
                str(detector),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()) as captured:
                score_feature_bundle.main()
            summary = json.loads(captured.getvalue())
            self.assertEqual(summary["mode"], "legacy_single")
            self.assertEqual(summary["feature_view"], "joint_representation")
            self.assertEqual(summary["feature_dim"], 4)
            with output.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = reader.fieldnames
            self.assertEqual(
                fieldnames,
                ["sample_id", "label_id", *score_feature_bundle.LEGACY_OUTPUT_FIELDS],
            )
            self.assertEqual(
                {row["feature_view"] for row in rows}, {"joint_representation"}
            )
            self.assertEqual({row["detector_mode"] for row in rows}, {"legacy_single"})

    def test_mode_must_be_explicit_and_match_detector_argument(self) -> None:
        base = [
            "score_feature_bundle.py",
            "--metadata",
            "metadata.csv",
            "--features",
            "features.npz",
            "--detector",
            "detector.npz",
            "--output",
            "scores.csv",
        ]
        with (
            patch.object(sys, "argv", base),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            score_feature_bundle.main()
        with (
            patch.object(sys, "argv", [base[0], "--mode", "dual_or", *base[1:]]),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            score_feature_bundle.main()


class _PrimitiveArtifact:
    def __init__(self, *, pooling: str, threshold: float, feature_dim: int = 2):
        self.pooling = pooling
        self.threshold = threshold
        self.feature_dim = feature_dim

    def score(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64)[:, 0]


class QualifiedDualScoringTests(unittest.TestCase):
    PROVENANCE = {
        "model_family": "test_family",
        "model_id": "test/model",
        "model_revision": "revision",
        "tokenizer_revision": "tokenizer",
        "preprocessing_sha256": "a" * 64,
        "layer": -1,
        "pooling": "text_image_tokens",
    }

    def _inputs(self, root: Path) -> tuple[Path, Path]:
        metadata = root / "metadata.csv"
        with metadata.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "label_id"])
            writer.writeheader()
            writer.writerows(
                [
                    {"sample_id": "text-block", "label_id": "1"},
                    {"sample_id": "both-review", "label_id": "0"},
                    {"sample_id": "both-allow", "label_id": "0"},
                ]
            )
        features = root / "features.npz"
        np.savez(
            features,
            sample_ids=np.asarray(["text-block", "both-review", "both-allow"]),
            text_embeddings=np.asarray(
                [[0.95, 0.0], [0.60, 0.0], [0.10, 0.0]]
            ),
            image_embeddings=np.asarray(
                [[0.10, 0.0], [0.75, 0.0], [0.20, 0.0]]
            ),
            attribution_features=np.zeros((3, 1)),
            feature_source=np.asarray(["synthetic"]),
            id_column=np.asarray(["sample_id"]),
            **{
                key: np.asarray([value], dtype=np.int64)
                if key == "layer"
                else np.asarray([value])
                for key, value in self.PROVENANCE.items()
            },
        )
        return metadata, features

    def _pair(self, root: Path, *, qualified: bool = True) -> SimpleNamespace:
        shared = {
            key: value
            for key, value in self.PROVENANCE.items()
            if key != "pooling"
        }
        shared["base_feature_dim"] = 2
        manifest = {
            "shared_provenance": shared,
            "cache_representation": {
                "pooling": "text_image_tokens",
                "feature_dim": 4,
                "ordering": ["text_tokens", "image_tokens"],
            },
            "qualification_evidence": {"passed": True} if qualified else None,
            "artifacts": {
                "text": {
                    "block_threshold": 0.9,
                    "review_threshold": 0.5,
                },
                "image": {
                    "block_threshold": 0.8,
                    "review_threshold": 0.7,
                },
            },
            "runtime_detector_identity_sha256": "b" * 64,
            "pair_identity_sha256": "c" * 64,
        }
        return SimpleNamespace(
            manifest=manifest,
            manifest_path=root / "detector_pair_manifest.json",
            text_artifact=_PrimitiveArtifact(
                pooling="text_tokens", threshold=0.9
            ),
            image_artifact=_PrimitiveArtifact(
                pooling="image_tokens", threshold=0.8
            ),
        )

    def test_qualified_pair_scores_primitive_views_and_composes_runtime_actions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, features = self._inputs(root)
            output = root / "scores.csv"
            pair = self._pair(root)
            argv = [
                "score_feature_bundle.py",
                "--mode",
                "dual_or",
                "--metadata",
                str(metadata),
                "--features",
                str(features),
                "--pair-manifest",
                str(pair.manifest_path),
                "--output",
                str(output),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    score_feature_bundle,
                    "load_detector_pair",
                    return_value=pair,
                ) as load_pair,
                redirect_stdout(io.StringIO()) as captured,
            ):
                score_feature_bundle.main()
            load_pair.assert_called_once_with(
                pair.manifest_path,
                expected_base_feature_dim=2,
            )
            summary = json.loads(captured.getvalue())
            self.assertEqual(summary["mode"], "dual_or")
            self.assertEqual(summary["primitive_feature_dim"], 2)
            self.assertEqual(summary["fused_feature_dim"], 4)
            self.assertEqual(
                summary["action_counts"],
                {"allow": 1, "block": 1, "review": 1},
            )
            with output.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = reader.fieldnames
            self.assertEqual(
                fieldnames,
                ["sample_id", "label_id", *score_feature_bundle.DUAL_OUTPUT_FIELDS],
            )
            self.assertEqual(
                [row["recommended_action"] for row in rows],
                ["block", "review", "allow"],
            )
            self.assertEqual(
                [row["decisive_head"] for row in rows],
                ["text", "image", "image"],
            )
            self.assertEqual(rows[0]["text_action"], "block")
            self.assertEqual(rows[0]["image_action"], "allow")
            self.assertEqual(float(rows[0]["risk_score"]), 0.95)
            self.assertEqual(float(rows[0]["detector_threshold"]), 0.9)
            self.assertEqual(float(rows[1]["risk_score"]), 0.75)
            self.assertEqual(float(rows[1]["review_threshold"]), 0.7)
            self.assertEqual(rows[1]["uncertain"], "True")
            self.assertEqual(rows[0]["runtime_detector_identity_sha256"], "b" * 64)
            self.assertEqual(rows[0]["pair_identity_sha256"], "c" * 64)

    def test_dual_scoring_rejects_unqualified_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, features = self._inputs(root)
            pair = self._pair(root, qualified=False)
            argv = [
                "score_feature_bundle.py",
                "--mode",
                "dual_or",
                "--metadata",
                str(metadata),
                "--features",
                str(features),
                "--pair-manifest",
                str(pair.manifest_path),
                "--output",
                str(root / "scores.csv"),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    score_feature_bundle,
                    "load_detector_pair",
                    return_value=pair,
                ),
                self.assertRaisesRegex(ValueError, "passing, bound qualification"),
            ):
                score_feature_bundle.main()

    def test_dual_bundle_requires_exact_provenance_and_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pair = self._pair(Path(temporary))
            provenance = dict(self.PROVENANCE)
            text = np.zeros((3, 2))
            image = np.zeros((3, 2))
            score_feature_bundle._validate_dual_bundle(
                bundle_provenance=provenance,
                text_features=text,
                image_features=image,
                manifest=pair.manifest,
            )
            changed = dict(provenance)
            changed["model_revision"] = "wrong"
            with self.assertRaisesRegex(ValueError, "model_revision"):
                score_feature_bundle._validate_dual_bundle(
                    bundle_provenance=changed,
                    text_features=text,
                    image_features=image,
                    manifest=pair.manifest,
                )
            changed_pooling = dict(provenance)
            changed_pooling["pooling"] = "image_tokens"
            with self.assertRaisesRegex(ValueError, "text_image_tokens"):
                score_feature_bundle._validate_dual_bundle(
                    bundle_provenance=changed_pooling,
                    text_features=text,
                    image_features=image,
                    manifest=pair.manifest,
                )
            pair.manifest["shared_provenance"]["base_feature_dim"] = 3
            with self.assertRaisesRegex(ValueError, "dimensions do not match"):
                score_feature_bundle._validate_dual_bundle(
                    bundle_provenance=provenance,
                    text_features=text,
                    image_features=image,
                    manifest=pair.manifest,
                )
            pair.manifest["shared_provenance"]["base_feature_dim"] = 2
            pair.manifest["cache_representation"]["ordering"] = [
                "image_tokens",
                "text_tokens",
            ]
            with self.assertRaisesRegex(ValueError, "not canonical"):
                score_feature_bundle._validate_dual_bundle(
                    bundle_provenance=provenance,
                    text_features=text,
                    image_features=image,
                    manifest=pair.manifest,
                )

    def test_decisive_head_uses_runtime_lexical_tie_break(self) -> None:
        decision = score_feature_bundle._compose_dual_decision(
            {
                "text": {
                    "risk_score": 0.6,
                    "block_threshold": 0.9,
                    "review_threshold": 0.5,
                    "pooling": "text_tokens",
                },
                "image": {
                    "risk_score": 0.6,
                    "block_threshold": 0.9,
                    "review_threshold": 0.5,
                    "pooling": "image_tokens",
                },
            }
        )
        self.assertEqual(decision["recommended_action"], "review")
        self.assertEqual(decision["decisive_head"], "image")
        self.assertEqual(decision["risk_score"], 0.6)
        self.assertEqual(decision["detector_threshold"], 0.9)
        self.assertEqual(decision["review_threshold"], 0.5)

    def test_invalid_scores_and_thresholds_fail_closed(self) -> None:
        base = {
            "text": {
                "risk_score": 0.6,
                "block_threshold": 0.9,
                "review_threshold": 0.5,
                "pooling": "text_tokens",
            },
            "image": {
                "risk_score": 0.6,
                "block_threshold": 0.8,
                "review_threshold": 0.4,
                "pooling": "image_tokens",
            },
        }
        mutations = (
            ("risk_score", float("nan"), "risk score"),
            ("risk_score", 1.01, "risk score"),
            ("block_threshold", 1.0, "block threshold"),
            ("review_threshold", 0.9, "review threshold"),
        )
        for field, value, expected in mutations:
            with self.subTest(field=field, value=value):
                heads = {name: dict(settings) for name, settings in base.items()}
                heads["text"][field] = value
                with self.assertRaisesRegex(ValueError, expected):
                    score_feature_bundle._compose_dual_decision(heads)

    def test_action_boundaries_are_inclusive(self) -> None:
        decision = score_feature_bundle._compose_dual_decision(
            {
                "text": {
                    "risk_score": 0.9,
                    "block_threshold": 0.9,
                    "review_threshold": 0.5,
                    "pooling": "text_tokens",
                },
                "image": {
                    "risk_score": 0.7,
                    "block_threshold": 0.8,
                    "review_threshold": 0.7,
                    "pooling": "image_tokens",
                },
            }
        )
        self.assertEqual(decision["heads"]["text"]["action"], "block")
        self.assertEqual(decision["heads"]["image"]["action"], "review")
        self.assertEqual(decision["recommended_action"], "block")
        self.assertEqual(decision["decisive_head"], "text")

    def test_decisive_head_uses_normalized_progress_not_raw_score(self) -> None:
        decision = score_feature_bundle._compose_dual_decision(
            {
                "text": {
                    "risk_score": 0.8,
                    "block_threshold": 0.99,
                    "review_threshold": 0.79,
                    "pooling": "text_tokens",
                },
                "image": {
                    "risk_score": 0.75,
                    "block_threshold": 0.8,
                    "review_threshold": 0.5,
                    "pooling": "image_tokens",
                },
            }
        )
        self.assertGreater(
            decision["heads"]["text"]["risk_score"],
            decision["heads"]["image"]["risk_score"],
        )
        self.assertEqual(decision["recommended_action"], "review")
        self.assertEqual(decision["decisive_head"], "image")
        self.assertEqual(decision["risk_score"], 0.75)

    def test_metadata_reserved_output_columns_are_rejected_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, features = self._inputs(root)
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "label_id", "risk_score"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"sample_id": "text-block", "label_id": "1", "risk_score": "spoof"},
                        {"sample_id": "both-review", "label_id": "0", "risk_score": "spoof"},
                        {"sample_id": "both-allow", "label_id": "0", "risk_score": "spoof"},
                    ]
                )
            pair = self._pair(root)
            cases = (
                ("dual_or", "--pair-manifest", pair.manifest_path),
                ("legacy_single", "--detector", root / "missing-detector.npz"),
            )
            for mode, detector_flag, detector_path in cases:
                with self.subTest(mode=mode):
                    argv = [
                        "score_feature_bundle.py",
                        "--mode",
                        mode,
                        "--metadata",
                        str(metadata),
                        "--features",
                        str(features),
                        detector_flag,
                        str(detector_path),
                        "--output",
                        str(root / f"{mode}.csv"),
                    ]
                    with (
                        patch.object(sys, "argv", argv),
                        patch.object(
                            score_feature_bundle,
                            "load_detector_pair",
                            return_value=pair,
                        ) as load_pair,
                        self.assertRaisesRegex(ValueError, "reserved score output fields"),
                    ):
                        score_feature_bundle.main()
                    load_pair.assert_not_called()


class FusedExtractionContractTests(unittest.TestCase):
    def test_fused_extraction_rejects_missing_text_or_image_before_model_load(self) -> None:
        cases = (
            ({"prompt_text": "caller text", "image_path": ""}, "one image_path"),
            ({"prompt_text": "", "image_path": "image.png"}, "non-empty prompt_text"),
        )
        for row, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                metadata = root / "metadata.csv"
                with metadata.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["sample_id", "prompt_text", "image_path"],
                    )
                    writer.writeheader()
                    writer.writerow({"sample_id": "sample", **row})
                argv = [
                    "extract_mllm_features.py",
                    "--model-family",
                    "qwen25_vl",
                    "--metadata",
                    str(metadata),
                    "--corpus-root",
                    str(root),
                    "--output-dir",
                    str(root / "features"),
                    "--model-revision",
                    "revision",
                    "--tokenizer-revision",
                    "tokenizer",
                    "--pooling",
                    "text_image_tokens",
                ]
                with patch.object(sys, "argv", argv):
                    with self.assertRaisesRegex(ValueError, expected):
                        extract_mllm_features.main()


if __name__ == "__main__":
    unittest.main()
