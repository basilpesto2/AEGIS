from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from aegis_research import bordair_evaluation as evaluation  # noqa: E402


def result_row(
    panel: str,
    label: int,
    action: str,
    sample_id: str,
) -> dict[str, object]:
    return {
        "panel": panel,
        "label_id": label,
        "recommended_action": action,
        "error_type": None,
        "risk_score": 0.9,
        "static_decision_matches": True,
        "traffic_mode": "shadow",
        "enforcement_action": "allow",
        "image_fingerprint_matches": True,
        "prompt_fingerprint_matches": True,
        "accepted": action == ("block" if label == 1 else "allow"),
        "sample_id": sample_id,
    }


class CountingProvider:
    def __init__(self, artifact, embedding: np.ndarray) -> None:
        self.calls = 0
        self.embedding = embedding
        for name in (
            "model_family",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "preprocessing_sha256",
            "layer",
            "pooling",
            "feature_dim",
        ):
            setattr(self, name, getattr(artifact, name))

    def embed(self, request) -> np.ndarray:
        self.calls += 1
        return self.embedding.copy()


def cache_fixture(directory: Path):
    image_path = directory / "sample.png"
    image_path.write_bytes(b"stable-image-bytes")
    metadata_path = directory / "panel.csv"
    metadata_path.write_text("stable-panel-metadata\n", encoding="utf-8")
    sample_id = "TI-00001"
    text = "fixed caller text"
    artifact = SimpleNamespace(
        source="head-v7",
        model_family="test-family",
        model_id="test-model",
        model_revision="test-revision",
        tokenizer_revision="test-tokenizer",
        preprocessing_sha256="a" * 64,
        layer=-1,
        pooling="text_image_tokens",
        feature_dim=4,
        threshold=0.8,
    )
    expectation = evaluation.FeatureCacheExpectation(
        sample_id=sample_id,
        panel="regression",
        caller_text_sha256=evaluation.sha256_text(text),
        image_sha256=evaluation.sha256_file(image_path),
        image_path="images_v7/final/TI-00001.png",
        resolved_image_path=image_path.resolve(),
        panel_metadata_sha256=evaluation.sha256_file(metadata_path),
        panel_metadata_path=metadata_path.resolve(),
        corpus_manifest_sha256="b" * 64,
    )
    request = SimpleNamespace(
        request_id=sample_id,
        text=text,
        image_paths=(str(image_path.resolve()),),
    )
    return artifact, expectation, request


class EvaluationV7HarnessTests(unittest.TestCase):
    def test_benign_block_gate_is_strictly_fewer_than_three(self) -> None:
        def panel(benign_blocks: int) -> list[dict[str, object]]:
            rows = [
                result_row("regression", 1, "block", f"m-{index}")
                for index in range(10)
            ]
            rows.extend(
                result_row(
                    "regression",
                    0,
                    "block" if index < benign_blocks else "allow",
                    f"b-{index}",
                )
                for index in range(10)
            )
            rows.extend(
                result_row("text_led", 1, "block", f"t-{index}")
                for index in range(10)
            )
            rows.extend(
                result_row("external_benign", 0, "allow", f"e-{index}")
                for index in range(20)
            )
            return rows

        self.assertTrue(
            evaluation.acceptance_report(panel(2), {"passed": True})["passed"]
        )
        report = evaluation.acceptance_report(panel(3), {"passed": True})
        self.assertFalse(report["passed"])
        self.assertFalse(
            report["checks"]["regression_fewer_than_three_benign_blocked"]
        )

    def test_internal_test_gate_requires_both_attack_channels(self) -> None:
        summary = {
            "test_metrics": {"false_negatives": 0},
            "test_subgroups": {
                "all_malicious": {"false_negatives": 0},
                "image_only": {"false_negatives": 0},
                "text_led": {"false_negatives": 0},
            },
            "acceptance": {
                "checks": {
                    "internal_test_no_false_negatives": True,
                    "internal_test_image_only_all_blocked": True,
                    "internal_test_text_led_all_blocked": True,
                }
            },
        }
        self.assertTrue(evaluation.internal_test_evidence(summary)["passed"])
        summary["test_subgroups"]["text_led"]["false_negatives"] = 1
        self.assertFalse(evaluation.internal_test_evidence(summary)["passed"])

    def test_external_benign_gate_is_strictly_fewer_than_three_of_twenty(self) -> None:
        def results(external_blocks: int) -> list[dict[str, object]]:
            rows = [
                result_row("regression", 1, "block", f"m-{index}")
                for index in range(10)
            ]
            rows.extend(
                result_row("regression", 0, "allow", f"b-{index}")
                for index in range(10)
            )
            rows.extend(
                result_row("text_led", 1, "block", f"t-{index}")
                for index in range(10)
            )
            rows.extend(
                result_row(
                    "external_benign",
                    0,
                    "block" if index < external_blocks else "allow",
                    f"e-{index}",
                )
                for index in range(20)
            )
            return rows

        self.assertTrue(
            evaluation.acceptance_report(results(2), {"passed": True})["passed"]
        )
        report = evaluation.acceptance_report(results(3), {"passed": True})
        self.assertFalse(report["passed"])
        self.assertFalse(
            report["checks"]["external_benign_fewer_than_three_blocked"]
        )

    def test_static_action_boundaries(self) -> None:
        self.assertEqual(
            evaluation.expected_decision_from_score(0.8, 0.8, 0.6),
            ("malicious", "block", False),
        )
        self.assertEqual(
            evaluation.expected_decision_from_score(0.6, 0.8, 0.6),
            ("benign", "review", True),
        )
        self.assertEqual(
            evaluation.expected_decision_from_score(0.59, 0.8, 0.6),
            ("benign", "allow", False),
        )

    def test_feature_cache_resumes_without_a_delegate_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, expectation, request = cache_fixture(root)
            delegate = CountingProvider(artifact, np.asarray([1.0, 2.0, 3.0, 4.0]))
            cache_root = root / "cache"
            first = evaluation.ResumableEvaluationFeatureProvider(
                delegate=delegate,
                artifact=artifact,
                expectations={expectation.sample_id: expectation},
                cache_root=cache_root,
            )
            expected = first.embed(request)
            self.assertEqual(delegate.calls, 1)
            self.assertEqual(first.cache_report()["writes"], 1)

            resumed = evaluation.ResumableEvaluationFeatureProvider(
                delegate=delegate,
                artifact=artifact,
                expectations={expectation.sample_id: expectation},
                cache_root=cache_root,
            )
            actual = resumed.embed(request)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(delegate.calls, 1)
            self.assertEqual(resumed.cache_report()["hits"], 1)
            cache_path = cache_root / "regression" / "TI-00001.npz"
            with np.load(cache_path, allow_pickle=False) as data:
                self.assertEqual(set(data.files), evaluation.FEATURE_CACHE_FIELDS)
                self.assertEqual(str(data["sample_id"][0]), expectation.sample_id)
                self.assertEqual(
                    str(data["caller_text_sha256"][0]),
                    expectation.caller_text_sha256,
                )
                self.assertEqual(
                    str(data["panel_metadata_sha256"][0]),
                    expectation.panel_metadata_sha256,
                )

    def test_feature_cache_rejects_corruption_and_provenance_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, expectation, request = cache_fixture(root)
            delegate = CountingProvider(artifact, np.asarray([1.0, 2.0, 3.0, 4.0]))
            cache_root = root / "cache"
            initial = evaluation.ResumableEvaluationFeatureProvider(
                delegate=delegate,
                artifact=artifact,
                expectations={expectation.sample_id: expectation},
                cache_root=cache_root,
            )
            initial.embed(request)
            cache_path = cache_root / "regression" / "TI-00001.npz"
            cache_path.write_bytes(b"truncated")

            recovered = evaluation.ResumableEvaluationFeatureProvider(
                delegate=delegate,
                artifact=artifact,
                expectations={expectation.sample_id: expectation},
                cache_root=cache_root,
            )
            recovered.embed(request)
            self.assertEqual(delegate.calls, 2)
            self.assertEqual(recovered.cache_report()["invalid_entries"], 1)
            self.assertEqual(recovered.cache_report()["writes"], 1)

            changed_artifact = SimpleNamespace(**vars(artifact))
            changed_artifact.model_revision = "different-revision"
            changed_delegate = CountingProvider(
                changed_artifact, np.asarray([4.0, 3.0, 2.0, 1.0])
            )
            changed = evaluation.ResumableEvaluationFeatureProvider(
                delegate=changed_delegate,
                artifact=changed_artifact,
                expectations={expectation.sample_id: expectation},
                cache_root=cache_root,
            )
            changed.embed(request)
            self.assertEqual(changed_delegate.calls, 1)
            self.assertEqual(changed.cache_report()["invalid_entries"], 1)

    def test_feature_cache_requires_exact_inputs_and_supports_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, expectation, request = cache_fixture(root)
            delegate = CountingProvider(artifact, np.asarray([1.0, 2.0, 3.0, 4.0]))
            cache_root = root / "cache"
            provider = evaluation.ResumableEvaluationFeatureProvider(
                delegate=delegate,
                artifact=artifact,
                expectations={expectation.sample_id: expectation},
                cache_root=cache_root,
            )
            provider.embed(request)

            changed_request = SimpleNamespace(
                request_id=request.request_id,
                text="changed caller text",
                image_paths=request.image_paths,
            )
            with self.assertRaisesRegex(ValueError, "Caller-text hash changed"):
                provider.embed(changed_request)
            self.assertEqual(delegate.calls, 1)

            image_path = Path(request.image_paths[0])
            original_image = image_path.read_bytes()
            image_path.write_bytes(b"changed-image-bytes")
            with self.assertRaisesRegex(ValueError, "Image bytes changed"):
                provider.embed(request)
            image_path.write_bytes(original_image)

            original_metadata = expectation.panel_metadata_path.read_bytes()
            expectation.panel_metadata_path.write_bytes(b"changed-panel-metadata")
            with self.assertRaisesRegex(ValueError, "Panel metadata changed"):
                provider.embed(request)
            expectation.panel_metadata_path.write_bytes(original_metadata)
            self.assertEqual(delegate.calls, 1)

            rebuilt = evaluation.ResumableEvaluationFeatureProvider(
                delegate=delegate,
                artifact=artifact,
                expectations={expectation.sample_id: expectation},
                cache_root=cache_root,
                rebuild=True,
            )
            rebuilt.embed(request)
            self.assertEqual(delegate.calls, 2)
            self.assertEqual(rebuilt.cache_report()["rebuilds"], 1)
            self.assertEqual(rebuilt.cache_report()["hits"], 0)


if __name__ == "__main__":
    unittest.main()
