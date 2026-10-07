from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
from threading import Lock, RLock
import time
import unittest
from unittest.mock import patch

from AEGIS.detector_artifact import load_detector_artifact
from AEGIS.detector_set import detector_set_identity_sha256
from AEGIS.audit import AuditLogConfig, PrivacySafeAuditLogger
from AEGIS.bounded_cache import BoundedTTLCache
from AEGIS.deployment_config import ServerConfig, load_deployment_config
from AEGIS.guardrail import GuardrailPolicy, GuardrailRequest, GuardrailRuntime
from AEGIS.isolated_service import (
    InferenceTimeoutError,
    InferenceWorkerError,
    ProcessIsolatedGuardrailService,
    _FailingProvider,
)
from AEGIS.http_server import GuardrailHTTPService
from AEGIS.gui_history import (
    GUIHistoryRetentionError,
    GUIHistoryStore,
    StoredImage,
)
from AEGIS.provenance import model_directory_fingerprint
from AEGIS.providers import (
    LlavaOnevisionGuardrailProvider,
    Qwen25VLGuardrailProvider,
)
from AEGIS.secret_validation import validate_distinct_secrets, validate_secret
from AEGIS.service import RequestLimits
from AEGIS.system_resources import (
    ResourceRequirements,
    ResourceSnapshot,
    evaluate_resource_requirements,
    inspect_model_cache,
)
from AEGIS.target_profiles import get_target_profile


def _abandoned_queue_teardown_probe() -> None:
    context = multiprocessing.get_context("spawn")
    request_queue = context.Queue(maxsize=1)
    request_queue.put(b"x" * (12 * 1024 * 1024), block=False)
    service = object.__new__(ProcessIsolatedGuardrailService)
    service._process = None
    service._request_queue = request_queue
    service._response_queue = None
    service._stop_worker()


class ProviderCacheKeyTests(unittest.TestCase):
    def test_qwen_key_uses_image_content_not_temporary_path(self) -> None:
        provider = Qwen25VLGuardrailProvider(environment_overrides=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"same-image")
            second.write_bytes(b"same-image")
            request_a = GuardrailRequest(
                text="prompt", image_paths=(str(first),), request_id="r1"
            )
            request_b = GuardrailRequest(
                text="prompt", image_paths=(str(second),), request_id="r2"
            )
            self.assertEqual(provider._cache_key(request_a), provider._cache_key(request_b))
            second.write_bytes(b"different-image")
            self.assertNotEqual(provider._cache_key(request_a), provider._cache_key(request_b))

    def test_llava_key_uses_image_content_not_temporary_path(self) -> None:
        provider = LlavaOnevisionGuardrailProvider(environment_overrides=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"same-image")
            second.write_bytes(b"same-image")
            request_a = GuardrailRequest(
                text="prompt", image_paths=(str(first),), request_id="r1"
            )
            request_b = GuardrailRequest(
                text="prompt", image_paths=(str(second),), request_id="r2"
            )
            self.assertEqual(provider._cache_key(request_a), provider._cache_key(request_b))


class IsolatedFailureProvenanceTests(unittest.TestCase):
    def test_live_worker_is_retained_and_blocks_replacement_after_kill_failure(self) -> None:
        class UnkillableProcess:
            pid = 606

            @staticmethod
            def is_alive() -> bool:
                return True

            @staticmethod
            def terminate() -> None:
                raise OSError("terminate failed")

            @staticmethod
            def kill() -> None:
                raise OSError("kill failed")

            @staticmethod
            def join(timeout=None) -> None:
                del timeout

        process = UnkillableProcess()
        service = object.__new__(ProcessIsolatedGuardrailService)
        service._request_lock = RLock()
        service._lifecycle_lock = Lock()
        service._closing = False
        service._readiness = {"ok": False}
        service._startup_warmup = None
        service._last_worker_error = None
        service._process = process
        service._request_queue = None
        service._response_queue = None

        self.assertFalse(service._stop_worker())
        self.assertIs(service._process, process)
        with patch(
            "AEGIS.isolated_service.multiprocessing.get_context"
        ) as get_context:
            with self.assertRaisesRegex(
                InferenceWorkerError,
                "replacement was not started",
            ):
                service._start_worker(initial=False)
        get_context.assert_not_called()
        self.assertIs(service._process, process)

    def test_fail_safe_preserves_timeout_error_after_layer_validation(self) -> None:
        artifact = load_detector_artifact(
            "models/aegis/llava05b_v7_dual_or/llava05b_text_head_v1.npz"
        )
        provider = _FailingProvider(
            artifact,
            InferenceTimeoutError("worker timed out"),
        )
        decision = GuardrailRuntime(
            artifact,
            policy=GuardrailPolicy(
                action_on_error="review",
                require_matching_provenance=True,
            ),
            include_error_details=True,
        ).evaluate_request(
            GuardrailRequest(text="prompt", image_paths=("unused.png",)),
            provider,
        )
        self.assertEqual(decision.verdict, "guardrail_error")
        self.assertEqual(decision.error_type, "InferenceTimeoutError")
        self.assertEqual(decision.error_detail, "worker timed out")

    def test_dead_worker_queue_teardown_does_not_wait_for_unread_payload(self) -> None:
        context = multiprocessing.get_context("spawn")
        process = context.Process(target=_abandoned_queue_teardown_probe)
        process.start()
        process.join(timeout=15.0)
        hung = process.is_alive()
        if hung:
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
        self.assertFalse(hung, "abandoned queue teardown blocked on its feeder thread")
        self.assertEqual(process.exitcode, 0)

    def test_transient_restart_failure_is_retried(self) -> None:
        service = object.__new__(ProcessIsolatedGuardrailService)
        service._closing = False
        service._last_worker_error = None
        service._readiness = {}
        with patch.object(
            service,
            "_start_worker",
            side_effect=[InferenceWorkerError("transient startup failure"), None],
        ) as start_worker:
            service._restart_worker()
        self.assertEqual(start_worker.call_count, 2)

    def test_stale_generation_cannot_invalidate_replacement_worker(self) -> None:
        class AliveProcess:
            pid = 404

            @staticmethod
            def is_alive() -> bool:
                return True

        service = object.__new__(ProcessIsolatedGuardrailService)
        service._request_lock = RLock()
        service._lifecycle_lock = Lock()
        service._restart_lock = Lock()
        service._worker_generation = 2
        service._closing = False
        service._readiness = {"ok": True, "checks": []}
        service._last_worker_error = None
        service._process = AliveProcess()
        service._request_queue = None
        service._response_queue = None
        replacement = service._process
        service._invalidate_worker("old generation failed", expected_generation=1)
        self.assertIs(service._process, replacement)
        self.assertTrue(service._readiness["ok"])
        self.assertIsNone(service._last_worker_error)

    def test_replacement_is_not_ready_until_startup_handshake(self) -> None:
        observed_ready = []

        class FakeQueue:
            def __init__(self, message=None) -> None:
                self.message = message

            def get(self, timeout=None):
                del timeout
                return self.message

            def cancel_join_thread(self) -> None:
                pass

            def close(self) -> None:
                pass

        class FakeProcess:
            pid = 505

            def __init__(self) -> None:
                self.alive = False

            def start(self) -> None:
                self.alive = True
                observed_ready.append(service.ready)

            def is_alive(self) -> bool:
                return self.alive

            def terminate(self) -> None:
                self.alive = False

            def kill(self) -> None:
                self.alive = False

            def join(self, timeout=None) -> None:
                del timeout

        request_queue = FakeQueue()
        response_queue = FakeQueue(
            {
                "kind": "ready",
                "readiness": {"ok": True, "checks": []},
                "warmup": {"status": "pass"},
            }
        )

        class FakeContext:
            def __init__(self) -> None:
                self.queues = [request_queue, response_queue]

            def Queue(self, maxsize):
                del maxsize
                return self.queues.pop(0)

            @staticmethod
            def Process(**_kwargs):
                return FakeProcess()

        service = object.__new__(ProcessIsolatedGuardrailService)
        service.artifact_path = Path("unused.npz")
        service.provider_spec = "unused:provider"
        service.provider_options = {}
        service.policy = GuardrailPolicy()
        service.limits = RequestLimits()
        service.warmup_payload = None
        service.target_profile = None
        service.input_modalities = ("image_text",)
        service.worker_startup_timeout_seconds = 1.0
        service._request_lock = RLock()
        service._lifecycle_lock = Lock()
        service._closing = False
        service._readiness = {"ok": True, "checks": ["stale"]}
        service._startup_warmup = {"status": "stale"}
        service._last_worker_error = "stale"
        service._process = None
        service._request_queue = None
        service._response_queue = None
        service._worker_generation = 0
        service._worker_restarts = 0
        with patch(
            "AEGIS.isolated_service.multiprocessing.get_context",
            return_value=FakeContext(),
        ):
            service._start_worker(initial=False)
        self.assertEqual(observed_ready, [False])
        self.assertTrue(service.ready)
        self.assertEqual(service._worker_generation, 1)
        self.assertEqual(service._worker_restarts, 1)
        service._stop_worker()

    def test_partial_worker_creation_failure_closes_created_queue(self) -> None:
        class FakeQueue:
            def __init__(self) -> None:
                self.cancelled = False
                self.closed = False

            def cancel_join_thread(self) -> None:
                self.cancelled = True

            def close(self) -> None:
                self.closed = True

        request_queue = FakeQueue()

        class FailingContext:
            def __init__(self) -> None:
                self.calls = 0

            def Queue(self, maxsize):
                del maxsize
                self.calls += 1
                if self.calls == 1:
                    return request_queue
                raise OSError("queue allocation failed")

        service = object.__new__(ProcessIsolatedGuardrailService)
        service.artifact_path = Path("unused.npz")
        service.provider_spec = "unused:provider"
        service.provider_options = {}
        service.policy = GuardrailPolicy()
        service.limits = RequestLimits()
        service.warmup_payload = None
        service.target_profile = None
        service.input_modalities = ("image_text",)
        service.worker_startup_timeout_seconds = 1.0
        service._request_lock = RLock()
        service._lifecycle_lock = Lock()
        service._closing = False
        service._readiness = {"ok": True}
        service._startup_warmup = None
        service._last_worker_error = None
        service._process = None
        service._request_queue = None
        service._response_queue = None
        service._worker_generation = 0
        service._worker_restarts = 0
        with patch(
            "AEGIS.isolated_service.multiprocessing.get_context",
            return_value=FailingContext(),
        ):
            with self.assertRaisesRegex(OSError, "queue allocation failed"):
                service._start_worker(initial=False)
        self.assertTrue(request_queue.cancelled)
        self.assertTrue(request_queue.closed)
        self.assertIsNone(service._request_queue)


class IntegrityTests(unittest.TestCase):
    def test_target_detector_hashes_match_files(self) -> None:
        for target in ("llava05b", "qwen25vl3b"):
            profile = get_target_profile(target)
            if profile.detector_heads:
                for head in profile.detector_heads:
                    digest = hashlib.sha256(
                        Path(head.detector).read_bytes()
                    ).hexdigest()
                    self.assertEqual(digest, head.detector_sha256)
                self.assertEqual(
                    profile.detector_identity_sha256,
                    detector_set_identity_sha256(
                        tuple(
                            (
                                head.name,
                                head.detector_sha256,
                                head.review_threshold,
                            )
                            for head in profile.detector_heads
                        )
                    ),
                )
            else:
                self.assertIsNotNone(profile.detector)
                digest = hashlib.sha256(
                    Path(str(profile.detector)).read_bytes()
                ).hexdigest()
                self.assertEqual(digest, profile.detector_sha256)
                self.assertEqual(
                    profile.detector_identity_sha256,
                    profile.detector_sha256,
                )

    def test_huggingface_snapshot_content_digest_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision = "a" * 40
            snapshot = root / "snapshots" / revision
            snapshot.mkdir(parents=True)
            for name, content in {
                "config.json": b"{}",
                "tokenizer.json": b"{}",
                "preprocessor_config.json": b"{}",
                "model.safetensors": b"weights",
            }.items():
                (snapshot / name).write_bytes(content)
            expected = str(model_directory_fingerprint(snapshot)["content_sha256"])
            good = inspect_model_cache(
                root,
                model_family="qwen25_vl",
                revision=revision,
                expected_content_sha256=expected,
            )
            bad = inspect_model_cache(
                root,
                model_family="qwen25_vl",
                revision=revision,
                expected_content_sha256="0" * 64,
            )
            self.assertTrue(good["ok"])
            self.assertFalse(bad["ok"])
            self.assertIn(
                "model runtime content matching configured SHA-256",
                bad["missing_files"],
            )

    def test_cuda_requirement_uses_free_not_total_vram(self) -> None:
        requirement = ResourceRequirements(min_cuda_device_memory_bytes=8)
        snapshot = ResourceSnapshot(
            total_physical_bytes=None,
            available_physical_bytes=None,
            total_virtual_bytes=None,
            available_virtual_bytes=None,
            disk_free_bytes=None,
            model_cache_bytes=None,
            cuda_devices=({"total_memory_bytes": 100, "free_memory_bytes": 7},),
        )
        report = evaluate_resource_requirements(requirement, snapshot)
        self.assertFalse(report["ok"])
        self.assertEqual(report["checks"][0]["name"], "cuda_device_available_memory")

    def test_audit_rotation_bounds_storage_across_runtime_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aegis-target.jsonl"
            config = AuditLogConfig(path=path, max_bytes=1024, backup_count=2)
            response = {
                "decisions": [
                    {
                        "action": "allow",
                        "recommended_action": "allow",
                        "verdict": "benign",
                        "reasons": ["bounded-session-test"],
                    }
                ]
            }
            for session_index in range(12):
                logger = PrivacySafeAuditLogger(
                    config,
                    context={"evidence_session_id": f"session-{session_index}"},
                )
                logger.record_response(
                    {"decisions": [dict(response["decisions"][0])]},
                    traffic_mode="shadow",
                    status=200,
                    duration_seconds=0.01,
                )
            retained = sorted(Path(directory).glob("aegis-target.jsonl*"))
            self.assertLessEqual(len(retained), 3)
            self.assertLessEqual(sum(item.stat().st_size for item in retained), 3 * 1024)


class ConfigurationTypeValidationTests(unittest.TestCase):
    def test_integer_limits_reject_booleans_floats_and_numeric_strings(self) -> None:
        invalid_values = (True, 1.5, "2")
        for value in invalid_values:
            with self.subTest(component="server", value=value):
                with self.assertRaises(ValueError):
                    ServerConfig(max_body_bytes=value)
            with self.subTest(component="request", value=value):
                with self.assertRaises(ValueError):
                    RequestLimits(max_batch_size=value)
            with self.subTest(component="resources", value=value):
                with self.assertRaises(ValueError):
                    ResourceRequirements(min_disk_free_bytes=value)
            with self.subTest(component="audit", value=value):
                with self.assertRaises(ValueError):
                    AuditLogConfig(max_bytes=value)
            with self.subTest(component="cache", value=value):
                with self.assertRaises(ValueError):
                    BoundedTTLCache(max_entries=value)

    def test_timeouts_and_thresholds_reject_non_numbers_and_nonfinite_values(self) -> None:
        invalid_values = (True, "1.0", float("nan"), float("inf"), 0.0)
        for value in invalid_values:
            with self.subTest(component="server", value=value):
                with self.assertRaises(ValueError):
                    ServerConfig(inference_timeout_seconds=value)
            with self.subTest(component="cache", value=value):
                with self.assertRaises(ValueError):
                    BoundedTTLCache(ttl_seconds=value)

        invalid_thresholds = (True, "0.4", float("nan"), float("inf"))
        for value in invalid_thresholds:
            with self.subTest(component="policy", value=value):
                with self.assertRaises(ValueError):
                    GuardrailPolicy(block_threshold=value)

    def test_boolean_flags_require_actual_booleans(self) -> None:
        with self.assertRaises(ValueError):
            RequestLimits(allow_local_image_paths=1)
        with self.assertRaises(ValueError):
            AuditLogConfig(fsync=1)
        with self.assertRaises(ValueError):
            GuardrailPolicy(require_matching_provenance=1)

    def test_policy_and_request_text_configuration_requires_exact_types(self) -> None:
        with self.assertRaises(ValueError):
            ServerConfig(host=123)
        with self.assertRaises(ValueError):
            ServerConfig(api_token_env="INVALID=NAME")
        with self.assertRaises(ValueError):
            ServerConfig(admin_token_env="invalid-name")
        with self.assertRaises(ValueError):
            GuardrailPolicy(fingerprint_key_env=123)
        with self.assertRaises(ValueError):
            GuardrailPolicy(fingerprint_key_env="invalid-name")
        with self.assertRaises(ValueError):
            RequestLimits(allowed_image_root=123, allow_local_image_paths=True)
        with self.assertRaises(ValueError):
            RequestLimits(allowed_image_root="", allow_local_image_paths=True)
        with self.assertRaises(ValueError):
            RequestLimits(allowed_image_media_types="image/png")
        with self.assertRaises(ValueError):
            RequestLimits(
                allowed_image_media_types=("image/png", "image/png")
            )

    def test_deployment_json_rejects_nonstandard_numeric_constants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "invalid.json"
            config_path.write_text(
                '{"schema_version":1,"name":"invalid","detector":"x",'
                '"provider":"x:y","policy":{},"server":{},"require_cuda":NaN}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Non-finite JSON number"):
                load_deployment_config(config_path)

    def test_deployment_json_rejects_coerced_scalar_identity_and_paths(self) -> None:
        base = {
            "schema_version": 1,
            "name": "valid-name",
            "detector": "detector.npz",
            "provider": "package.module:Provider",
            "policy": {},
            "server": {},
        }
        mutations = (
            {"schema_version": True},
            {"name": 123},
            {"detector": 123},
            {"provider": 123},
            {"base_dir": 123},
            {"cache_dir": 123},
            {"warmup_request_json": 123},
            {"target_profile": 123},
            {"traffic_mode": 123},
            {"audit": {"path": 123}},
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "invalid.json"
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    payload = dict(base)
                    payload.update(mutation)
                    config_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_deployment_config(config_path)


class GUIHistoryRetentionTests(unittest.TestCase):
    def test_expired_active_pending_record_survives_until_terminal_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            store = GUIHistoryStore(
                database,
                max_records=1,
                max_age_days=0.000001,
                max_bytes=1024 * 1024,
            )
            record_id = store.create_pending(
                text="long-running inference",
                request_id="active-request",
                metadata={},
                images=(),
            )
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "UPDATE gui_history SET created_at = ?, updated_at = ? "
                    "WHERE record_id = ?",
                    ("2000-01-01T00:00:00.000Z",) * 2 + (record_id,),
                )
                connection.commit()
            finally:
                connection.close()

            self.assertIsNotNone(store.get(record_id))
            self.assertEqual(store.summary()["status_counts"]["pending"], 1)
            store.fail(record_id, {"error": {"message": "terminal"}}, 1.0)
            self.assertIsNone(store.get(record_id))

    def test_retention_configuration_requires_exact_numeric_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            for invalid in (True, 1.5, "2"):
                with self.subTest(field="max_records", value=invalid):
                    with self.assertRaises(ValueError):
                        GUIHistoryStore(path, max_records=invalid)
                with self.subTest(field="max_bytes", value=invalid):
                    with self.assertRaises(ValueError):
                        GUIHistoryStore(path, max_bytes=invalid)
            for invalid in (True, "2.0", float("nan"), float("inf"), 0.0):
                with self.subTest(field="max_age_days", value=invalid):
                    with self.assertRaises(ValueError):
                        GUIHistoryStore(path, max_age_days=invalid)

    def test_record_cap_prunes_old_completed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GUIHistoryStore(
                Path(directory) / "history.sqlite3",
                max_records=2,
                max_age_days=30,
                max_bytes=1024 * 1024,
            )
            identifiers = []
            for index in range(3):
                record_id = store.create_pending(
                    text=f"prompt-{index}",
                    request_id=f"request-{index}",
                    metadata={},
                    images=(),
                )
                store.fail(record_id, {"error": {"message": "test"}}, 1.0)
                identifiers.append(record_id)
            self.assertEqual(store.summary()["total"], 2)
            self.assertIsNone(store.get(identifiers[0]))

    def test_expired_record_is_pruned_before_read_without_another_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GUIHistoryStore(
                Path(directory) / "history.sqlite3",
                max_records=10,
                max_age_days=0.000001,
                max_bytes=1024 * 1024,
            )
            record_id = store.create_pending(
                text="short-lived prompt",
                request_id="request",
                metadata={},
                images=(StoredImage("sample.png", "image/png", b"image"),),
            )
            store.fail(record_id, {"error": {"message": "test"}}, 1.0)
            time.sleep(0.15)
            self.assertIsNone(store.get(record_id))
            self.assertIsNone(store.get_image(record_id, 0))
            self.assertEqual(store.summary()["total"], 0)

    def test_active_record_over_budget_raises_named_retention_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GUIHistoryStore(
                Path(directory) / "history.sqlite3",
                max_records=1,
                max_age_days=30,
                max_bytes=32,
            )
            with self.assertRaises(GUIHistoryRetentionError):
                store.create_pending(
                    text="payload larger than the configured logical byte budget",
                    request_id="request",
                    metadata={},
                    images=(),
                )

    def test_restart_marks_pending_rows_interrupted_before_retention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            store = GUIHistoryStore(
                database,
                max_records=10,
                max_age_days=30,
                max_bytes=1024 * 1024,
            )
            interrupted_id = store.create_pending(
                text="interrupted prompt",
                request_id="interrupted-request",
                metadata={},
                images=(
                    StoredImage(
                        "large.png",
                        "image/png",
                        b"x" * (128 * 1024),
                    ),
                ),
            )

            restarted = GUIHistoryStore(
                database,
                max_records=10,
                max_age_days=30,
                max_bytes=1024 * 1024,
            )
            recovered = restarted.get(interrupted_id)
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered["status"], "error")
            self.assertEqual(recovered["error"]["error"]["code"], "gui_interrupted")

            over_budget_restart = GUIHistoryStore(
                database,
                max_records=10,
                max_age_days=30,
                max_bytes=4096,
            )
            self.assertIsNone(over_budget_restart.get(interrupted_id))
            replacement_id = over_budget_restart.create_pending(
                text="small replacement",
                request_id="replacement-request",
                metadata={},
                images=(),
            )
            self.assertIsNotNone(over_budget_restart.get(replacement_id))

    def test_clear_compacts_raw_prompt_and_image_content(self) -> None:
        marker = b"unique-sensitive-marker-704235"
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            store = GUIHistoryStore(database)
            record_id = store.create_pending(
                text=marker.decode("ascii"),
                request_id="request",
                metadata={},
                images=(StoredImage("sample.png", "image/png", marker),),
            )
            store.fail(record_id, {"error": {"message": "test"}}, 1.0)
            result = store.clear_all()
            self.assertEqual(result["deleted_records"], 1)
            self.assertEqual(store.summary()["total"], 0)
            self.assertNotIn(marker, database.read_bytes())
            wal = Path(f"{database}-wal")
            if wal.exists():
                self.assertNotIn(marker, wal.read_bytes())


class SecretValidationTests(unittest.TestCase):
    def test_runtime_service_reprs_exclude_bearer_fields(self) -> None:
        for service_class in (GuardrailHTTPService, ProcessIsolatedGuardrailService):
            with self.subTest(service=service_class.__name__):
                self.assertFalse(service_class.__dataclass_fields__["api_token"].repr)
                self.assertFalse(service_class.__dataclass_fields__["admin_token"].repr)

    def test_placeholder_weak_and_reused_secrets_are_rejected(self) -> None:
        _, placeholder = validate_secret(
            "replace-with-a-private-random-token-value",
            name="API token",
            required=True,
        )
        _, short = validate_secret("too-short", name="API token", required=True)
        self.assertTrue(placeholder)
        self.assertTrue(short)
        strong = "p5c_8XJvA1-Ns7Wq2kMz4Hr9Ty6Ld0Be"
        _, valid = validate_secret(strong, name="API token", required=True)
        self.assertFalse(valid)
        reused = validate_distinct_secrets({"API token": strong, "HMAC key": strong})
        self.assertTrue(reused)

    def test_character_dominated_secret_is_rejected(self) -> None:
        weak = "a" * 28 + "bcde"
        _, problems = validate_secret(weak, name="API token", required=True)
        self.assertTrue(problems)
        self.assertTrue(any("random value" in problem for problem in problems))

    def test_runtime_rejects_reused_fingerprint_and_api_secrets(self) -> None:
        artifact = load_detector_artifact(
            "models/aegis/llava05b_v7_dual_or/llava05b_text_head_v1.npz"
        )
        strong = "p5c_8XJvA1-Ns7Wq2kMz4Hr9Ty6Ld0Be"
        with patch.dict(os.environ, {"AEGIS_TEST_FINGERPRINT": strong}):
            with self.assertRaisesRegex(ValueError, "must be different secrets"):
                GuardrailHTTPService(
                    artifact=artifact,
                    provider=_FailingProvider(artifact, RuntimeError("unused")),
                    policy=GuardrailPolicy(
                        fingerprint_key_env="AEGIS_TEST_FINGERPRINT"
                    ),
                    api_token=strong,
                )

    def test_runtime_rejects_weak_fingerprint_secret(self) -> None:
        artifact = load_detector_artifact(
            "models/aegis/llava05b_v7_dual_or/llava05b_text_head_v1.npz"
        )
        with patch.dict(
            os.environ,
            {"AEGIS_TEST_FINGERPRINT": "replace-with-a-private-random-value"},
        ):
            with self.assertRaisesRegex(ValueError, "placeholder"):
                GuardrailHTTPService(
                    artifact=artifact,
                    provider=_FailingProvider(artifact, RuntimeError("unused")),
                    policy=GuardrailPolicy(
                        fingerprint_key_env="AEGIS_TEST_FINGERPRINT"
                    ),
                )

    def test_unicode_bearer_secret_is_rejected_and_presented_value_is_safe(self) -> None:
        unicode_secret = "éø漢字" * 8
        _, problems = validate_secret(
            unicode_secret,
            name="API token",
            required=True,
        )
        self.assertTrue(any("visible ASCII" in problem for problem in problems))

        artifact = load_detector_artifact(
            "models/aegis/llava05b_v7_dual_or/llava05b_text_head_v1.npz"
        )
        service = GuardrailHTTPService(
            artifact=artifact,
            provider=_FailingProvider(artifact, RuntimeError("unused")),
            policy=GuardrailPolicy(),
            api_token="p5c_8XJvA1-Ns7Wq2kMz4Hr9Ty6Ld0Be",
        )
        self.assertFalse(service.authorize(f"Bearer {unicode_secret}"))


if __name__ == "__main__":
    unittest.main()
