from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from AEGIS.audit import AuditLogConfig, PrivacySafeAuditLogger
from AEGIS.deployment_config import (
    build_service_from_config,
    load_configured_detector,
    load_deployment_config,
)
from AEGIS.detector_artifact import DetectorArtifact, save_detector_artifact
from AEGIS.detector_set import (
    DetectorHeadConfig,
    OrDetector,
    detector_set_identity_sha256,
    load_detector_definition,
    load_or_detector,
)
from AEGIS.guardrail import GuardrailPolicy, GuardrailRequest, GuardrailRuntime
from AEGIS.http_server import GuardrailHTTPService, public_readiness_report
from AEGIS.isolated_service import (
    InferenceWorkerError,
    ProcessIsolatedGuardrailService,
)
from AEGIS.logistic import LogisticRegressionNumpy


def _artifact(pooling: str, *, threshold: float, bias: float = 0.0) -> DetectorArtifact:
    classifier = LogisticRegressionNumpy()
    classifier.weights_ = np.asarray([1.0, 0.0], dtype=np.float64)
    classifier.bias_ = bias
    classifier.mean_ = np.zeros(2, dtype=np.float64)
    classifier.scale_ = np.ones(2, dtype=np.float64)
    return DetectorArtifact(
        classifier=classifier,
        threshold=threshold,
        uncertainty_margin=0.01,
        model_family="generic",
        model_id="fixture-model",
        model_revision="fixture-revision",
        tokenizer_revision="fixture-tokenizer",
        preprocessing_sha256="a" * 64,
        layer=-1,
        pooling=pooling,
        source=f"fixture-{pooling}",
    )


class _Provider:
    model_family = "generic"
    model_id = "fixture-model"
    model_revision = "fixture-revision"
    tokenizer_revision = "fixture-tokenizer"
    preprocessing_sha256 = "a" * 64
    layer = -1
    pooling = "text_image_tokens"
    feature_dim = 4

    def __init__(self, vector=(0.0, 0.0, 0.0, 0.0)) -> None:
        self.vector = np.asarray(vector, dtype=np.float64)
        self.calls = 0

    def embed(self, request: GuardrailRequest) -> np.ndarray:
        del request
        self.calls += 1
        return self.vector.copy()


class _SingleProvider(_Provider):
    pooling = "text_tokens"
    feature_dim = 2

    def __init__(self) -> None:
        super().__init__((0.0, 0.0))


class DualDetectorRuntimeTests(unittest.TestCase):
    def _detector(self, root: Path) -> OrDetector:
        text = root / "text.npz"
        image = root / "image.npz"
        save_detector_artifact(text, _artifact("text_tokens", threshold=0.70))
        save_detector_artifact(image, _artifact("image_tokens", threshold=0.80))
        return load_or_detector(
            (
                DetectorHeadConfig("text", text, 0.55),
                DetectorHeadConfig("image", image, 0.60),
            )
        )

    def test_or_runtime_uses_one_fused_embedding_and_exposes_heads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = self._detector(Path(directory))
            provider = _Provider([2.0, 0.0, -2.0, 0.0])
            decision = GuardrailRuntime(
                detector,
                policy=GuardrailPolicy(require_matching_provenance=True),
            ).evaluate_request(
                GuardrailRequest(text="prompt", image_paths=("unused.png",)),
                provider,
            )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(decision.action, "block")
        self.assertEqual(decision.decisive_head, "text")
        self.assertEqual(decision.detector_mode, "or")
        self.assertEqual(len(decision.head_decisions), 2)
        self.assertEqual(decision.head_decisions[0]["recommended_action"], "block")
        self.assertEqual(decision.head_decisions[1]["recommended_action"], "allow")

    def test_legacy_single_decision_does_not_emit_dual_fields(self) -> None:
        decision = GuardrailRuntime(_artifact("text_tokens", threshold=0.70)).evaluate_features(
            np.asarray([[0.0, 0.0]], dtype=np.float64)
        )[0].to_dict()
        self.assertNotIn("detector_mode", decision)
        self.assertNotIn("decisive_head", decision)
        self.assertNotIn("head_decisions", decision)
        self.assertEqual(
            set(decision),
            {
                "action", "detector_source", "error_detail", "error_type",
                "fingerprint_algorithm", "image_hmac_sha256", "image_sha256",
                "modality", "model_family", "model_id", "pooling",
                "prompt_hmac_sha256", "prompt_sha256", "reasons", "request_id",
                "review_threshold", "risk_score", "sample_id", "schema_version",
                "threshold", "uncertain", "verdict",
            },
        )

    def test_or_action_precedence_and_decisive_threshold_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = self._detector(Path(directory))
            runtime = GuardrailRuntime(detector)
            review = runtime.evaluate_features(
                np.asarray([[0.4, 0.0, 0.0, 0.0]], dtype=np.float64)
            )[0]
            allowed = runtime.evaluate_features(
                np.asarray([[-2.0, 0.0, -2.0, 0.0]], dtype=np.float64)
            )[0]
        self.assertEqual(review.action, "review")
        self.assertEqual(review.decisive_head, "text")
        self.assertEqual(review.threshold, 0.70)
        self.assertEqual(review.review_threshold, 0.55)
        self.assertEqual(allowed.action, "allow")

    def test_identity_is_order_independent_and_threshold_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector = self._detector(root)
            reversed_detector = load_or_detector(
                tuple(
                    reversed(
                        (
                            DetectorHeadConfig("text", root / "text.npz", 0.55),
                            DetectorHeadConfig("image", root / "image.npz", 0.60),
                        )
                    )
                )
            )
            changed = load_or_detector(
                (
                    DetectorHeadConfig("text", root / "text.npz", 0.54),
                    DetectorHeadConfig("image", root / "image.npz", 0.60),
                )
            )
        self.assertEqual(detector.identity_sha256, reversed_detector.identity_sha256)
        self.assertNotEqual(detector.identity_sha256, changed.identity_sha256)

    def test_identity_rejects_noncanonical_artifact_hash(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical lowercase"):
            detector_set_identity_sha256(
                (("text", "A" * 64, 0.55), ("image", "b" * 64, 0.60))
            )

    def test_wrong_pooling_and_incompatible_provenance_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = root / "text.npz"
            wrong_image = root / "wrong-image.npz"
            save_detector_artifact(text, _artifact("text_tokens", threshold=0.70))
            save_detector_artifact(
                wrong_image, _artifact("text_tokens", threshold=0.80)
            )
            with self.assertRaisesRegex(ValueError, "requires pooling"):
                load_or_detector(
                    (
                        DetectorHeadConfig("text", text, 0.55),
                        DetectorHeadConfig("image", wrong_image, 0.60),
                    )
                )

            incompatible_image = root / "incompatible-image.npz"
            artifact = _artifact("image_tokens", threshold=0.80)
            object.__setattr__(artifact, "model_revision", "other-revision")
            save_detector_artifact(incompatible_image, artifact)
            with self.assertRaisesRegex(ValueError, "incompatible embedding provenance"):
                load_or_detector(
                    (
                        DetectorHeadConfig("text", text, 0.55),
                        DetectorHeadConfig("image", incompatible_image, 0.60),
                    )
                )

    def test_artifact_identity_is_bound_to_loaded_byte_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = root / "text.npz"
            replacement = root / "replacement.npz"
            image = root / "image.npz"
            save_detector_artifact(text, _artifact("text_tokens", threshold=0.70))
            save_detector_artifact(
                replacement, _artifact("text_tokens", threshold=0.75, bias=1.0)
            )
            save_detector_artifact(image, _artifact("image_tokens", threshold=0.80))
            original_bytes = text.read_bytes()
            replacement_bytes = replacement.read_bytes()
            original_read_bytes = Path.read_bytes
            replaced = False

            def replace_after_snapshot(path: Path) -> bytes:
                nonlocal replaced
                payload = original_read_bytes(path)
                if path == text and not replaced:
                    path.write_bytes(replacement_bytes)
                    replaced = True
                return payload

            with mock.patch.object(Path, "read_bytes", replace_after_snapshot):
                detector = load_or_detector(
                    (
                        DetectorHeadConfig("text", text, 0.55),
                        DetectorHeadConfig("image", image, 0.60),
                    )
                )

        text_head = detector.ordered_heads[0]
        self.assertEqual(
            text_head.artifact_sha256, hashlib.sha256(original_bytes).hexdigest()
        )
        self.assertEqual(text_head.artifact.threshold, 0.70)
        self.assertNotEqual(
            text_head.artifact_sha256, hashlib.sha256(replacement_bytes).hexdigest()
        )

    def test_dual_rejects_global_threshold_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = self._detector(Path(directory))
            with self.assertRaisesRegex(ValueError, "per-head thresholds"):
                GuardrailRuntime(detector, GuardrailPolicy(block_threshold=0.5))

    def test_schema_two_loads_dual_and_schema_one_stays_single(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._detector(root)
            common = {
                "name": "fixture",
                "provider": "tests.test_dual_detector_runtime:_Provider",
                "policy": {},
                "server": {},
            }
            dual_path = root / "dual.json"
            dual_path.write_text(
                json.dumps(
                    {
                        **common,
                        "schema_version": 2,
                        "detector": {
                            "mode": "or",
                            "heads": [
                                {"name": "text", "artifact": "text.npz", "review_threshold": 0.55},
                                {"name": "image", "artifact": "image.npz", "review_threshold": 0.60},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            single_path = root / "single.json"
            single_path.write_text(
                json.dumps({**common, "schema_version": 1, "detector": "text.npz"}),
                encoding="utf-8",
            )
            dual = load_deployment_config(dual_path)
            single = load_deployment_config(single_path)
            self.assertIsInstance(load_configured_detector(dual), OrDetector)
            self.assertFalse(single.detector_heads)
            self.assertEqual(single.detector_path.name, "text.npz")
            service, startup = build_service_from_config(dual)
            self.assertTrue(service.ready)
            self.assertIsNone(startup["warmup"])

    def test_schema_two_rejects_boolean_review_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "dual.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "name": "fixture",
                        "provider": "tests.test_dual_detector_runtime:_Provider",
                        "detector": {
                            "mode": "or",
                            "heads": [
                                {
                                    "name": "text",
                                    "artifact": "text.npz",
                                    "review_threshold": True,
                                },
                                {
                                    "name": "image",
                                    "artifact": "image.npz",
                                    "review_threshold": 0.60,
                                },
                            ],
                        },
                        "policy": {},
                        "server": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be finite"):
                load_deployment_config(config_path)

    def test_dual_status_is_authenticated_detail_not_public_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = self._detector(Path(directory))
            service = GuardrailHTTPService(
                artifact=detector,
                provider=_Provider([0.0, 0.0, 0.0, 0.0]),
                policy=GuardrailPolicy(require_matching_provenance=True),
            )
            detailed = service.readiness_report()
            public = public_readiness_report(detailed)
        self.assertEqual(detailed["detector_mode"], "or")
        self.assertEqual(detailed["detector_identity_sha256"], detector.identity_sha256)
        self.assertEqual(len(detailed["detector_heads"]), 2)
        self.assertNotIn("detector_identity_sha256", public)
        self.assertNotIn("detector_heads", public)

    def test_audit_identity_distinguishes_dual_composite_from_single_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector = self._detector(root)
            dual_path = root / "dual-audit.jsonl"
            dual_logger = PrivacySafeAuditLogger(
                AuditLogConfig(path=dual_path),
                context={
                    "detector_sha256": None,
                    "detector_identity_sha256": detector.identity_sha256,
                },
            )
            original_dual_identity = detector.identity_sha256
            save_detector_artifact(
                root / "text.npz",
                _artifact("text_tokens", threshold=0.75, bias=1.0),
            )
            mutated_detector = load_or_detector(
                (
                    DetectorHeadConfig("text", root / "text.npz", 0.55),
                    DetectorHeadConfig("image", root / "image.npz", 0.60),
                )
            )
            self.assertNotEqual(
                mutated_detector.identity_sha256,
                original_dual_identity,
            )
            response = {
                "decisions": [
                    {
                        "action": "allow",
                        "recommended_action": "allow",
                        "verdict": "benign",
                        "reasons": ["dual identity test"],
                    }
                ]
            }
            dual_logger.record_response(
                response,
                traffic_mode="shadow",
                status=200,
                duration_seconds=0.01,
            )
            dual_event = json.loads(dual_path.read_text(encoding="utf-8"))

            single_digest = "c" * 64
            single_path = root / "single-audit.jsonl"
            single_logger = PrivacySafeAuditLogger(
                AuditLogConfig(path=single_path),
                context={
                    "detector_sha256": single_digest,
                    "detector_identity_sha256": single_digest,
                },
            )
            single_logger.record_response(
                response,
                traffic_mode="shadow",
                status=200,
                duration_seconds=0.01,
            )
            single_event = json.loads(single_path.read_text(encoding="utf-8"))

        self.assertIsNone(dual_event["detector_sha256"])
        self.assertEqual(
            dual_event["detector_identity_sha256"], original_dual_identity
        )
        self.assertEqual(single_event["detector_sha256"], single_digest)
        self.assertNotIn("detector_identity_sha256", single_event)

    def test_process_isolated_dual_startup_reports_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector = self._detector(root)
            service = ProcessIsolatedGuardrailService(
                artifact_path=None,
                detector_heads=(
                    DetectorHeadConfig("text", root / "text.npz", 0.55),
                    DetectorHeadConfig("image", root / "image.npz", 0.60),
                ),
                provider_spec="tests.test_dual_detector_runtime:_Provider",
                provider_options={},
                policy=GuardrailPolicy(require_matching_provenance=True),
                worker_startup_timeout_seconds=20.0,
            )
            try:
                status = service.readiness_report()
                self.assertTrue(status["ok"])
                self.assertEqual(
                    status["detector_identity_sha256"], detector.identity_sha256
                )
                response = service.evaluate({"text": "prompt"})
                decision = response["decisions"][0]
                self.assertEqual(decision["detector_mode"], "or")
                self.assertIn(decision["decisive_head"], {"text", "image"})
                self.assertEqual(len(decision["head_decisions"]), 2)
            finally:
                service.close()

    def test_process_isolated_schema_one_startup_reports_snapshot_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "single.npz"
            save_detector_artifact(
                artifact_path, _artifact("text_tokens", threshold=0.70)
            )
            expected_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            service = ProcessIsolatedGuardrailService(
                artifact_path=artifact_path,
                provider_spec="tests.test_dual_detector_runtime:_SingleProvider",
                provider_options={},
                policy=GuardrailPolicy(require_matching_provenance=True),
                worker_startup_timeout_seconds=20.0,
            )
            try:
                status = service.readiness_report()
                self.assertTrue(status["ok"])
                self.assertEqual(status["detector_sha256"], expected_digest)
                decision = service.evaluate({"text": "prompt"})["decisions"][0]
                self.assertNotIn("detector_mode", decision)
                self.assertNotIn("head_decisions", decision)
            finally:
                service.close()

    def test_process_startup_rejects_single_and_dual_snapshot_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single_path = root / "single.npz"
            save_detector_artifact(
                single_path,
                _artifact("text_tokens", threshold=0.70),
            )
            single_snapshot = load_detector_definition(single_path)
            save_detector_artifact(
                single_path,
                _artifact("text_tokens", threshold=0.75, bias=1.0),
            )
            with self.assertRaisesRegex(
                InferenceWorkerError,
                "differs from the parent's validated detector snapshot",
            ):
                ProcessIsolatedGuardrailService(
                    artifact_path=single_path,
                    artifact_snapshot=single_snapshot,
                    provider_spec="tests.test_dual_detector_runtime:_SingleProvider",
                    provider_options={},
                    policy=GuardrailPolicy(require_matching_provenance=True),
                    worker_startup_timeout_seconds=20.0,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dual_snapshot = self._detector(root)
            save_detector_artifact(
                root / "text.npz",
                _artifact("text_tokens", threshold=0.75, bias=1.0),
            )
            with self.assertRaisesRegex(
                InferenceWorkerError,
                "differs from the parent's validated detector snapshot",
            ):
                ProcessIsolatedGuardrailService(
                    artifact_path=None,
                    artifact_snapshot=dual_snapshot,
                    detector_heads=(
                        DetectorHeadConfig("text", root / "text.npz", 0.55),
                        DetectorHeadConfig("image", root / "image.npz", 0.60),
                    ),
                    provider_spec="tests.test_dual_detector_runtime:_Provider",
                    provider_options={},
                    policy=GuardrailPolicy(require_matching_provenance=True),
                    worker_startup_timeout_seconds=20.0,
                )


if __name__ == "__main__":
    unittest.main()
