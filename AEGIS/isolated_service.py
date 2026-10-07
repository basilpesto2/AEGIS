from __future__ import annotations

from dataclasses import dataclass, field
import math
import multiprocessing
from pathlib import Path
from queue import Empty, Full
from threading import BoundedSemaphore, Lock, RLock, Thread
from typing import Any
import time
import uuid

import numpy as np

from AEGIS.audit import (
    PrivacySafeAuditLogger,
    apply_traffic_mode,
    validate_traffic_mode,
)
from AEGIS.detector_artifact import DetectorArtifact
from AEGIS.detector_set import DetectorHeadConfig, OrDetector, load_detector_definition
from AEGIS.guardrail import GuardrailPolicy, GuardrailRequest, resolve_fingerprint_key
from AEGIS.http_server import (
    GuardrailHTTPService,
    ServiceBusyError,
    ServiceMetrics,
)
from AEGIS.provider_contract import load_provider
from AEGIS.service import (
    REQUEST_MODALITIES,
    RequestLimits,
    evaluate_request_payload,
    normalize_input_modalities,
)
from AEGIS.secret_validation import (
    secret_matches,
    validate_distinct_secrets,
    validate_secret,
)


_WORKER_RESTART_INITIAL_BACKOFF_SECONDS = 1.0
_WORKER_RESTART_MAX_BACKOFF_SECONDS = 30.0


class InferenceTimeoutError(TimeoutError):
    pass


class InferenceWorkerError(RuntimeError):
    pass


@dataclass
class ProcessIsolatedGuardrailService:
    artifact_path: Path | None
    provider_spec: str
    provider_options: dict[str, object]
    policy: GuardrailPolicy
    limits: RequestLimits = field(default_factory=RequestLimits)
    api_token: str | None = field(default=None, repr=False)
    admin_token: str | None = field(default=None, repr=False)
    max_body_bytes: int = 12 * 1024 * 1024
    max_concurrent_requests: int = 1
    inference_timeout_seconds: float = 60.0
    worker_startup_timeout_seconds: float = 600.0
    warmup_payload: dict[str, Any] | None = None
    traffic_mode: str = "enforce"
    audit_logger: PrivacySafeAuditLogger | None = None
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)
    target_profile: str | None = None
    input_modalities: tuple[str, ...] | None = None
    detector_heads: tuple[DetectorHeadConfig, ...] = ()
    artifact_snapshot: DetectorArtifact | OrDetector | None = field(
        default=None, repr=False
    )
    _artifact: DetectorArtifact | OrDetector = field(init=False, repr=False)
    _capacity: BoundedSemaphore = field(init=False, repr=False)
    _request_lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _lifecycle_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _restart_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _traffic_mode_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _process: Any = field(default=None, init=False, repr=False)
    _request_queue: Any = field(default=None, init=False, repr=False)
    _response_queue: Any = field(default=None, init=False, repr=False)
    _readiness: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _startup_warmup: dict[str, object] | None = field(default=None, init=False, repr=False)
    _worker_generation: int = field(default=0, init=False)
    _worker_restarts: int = field(default=0, init=False)
    _last_worker_error: str | None = field(default=None, init=False)
    _closing: bool = field(default=False, init=False, repr=False)
    _restart_thread: Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_body_bytes, bool)
            or not isinstance(self.max_body_bytes, int)
            or self.max_body_bytes <= 0
        ):
            raise ValueError("max_body_bytes must be positive.")
        if isinstance(self.max_concurrent_requests, bool) or self.max_concurrent_requests != 1:
            raise ValueError(
                "Process-isolated inference currently requires "
                "server.max_concurrent_requests=1."
            )
        if (
            isinstance(self.inference_timeout_seconds, bool)
            or not isinstance(self.inference_timeout_seconds, (int, float))
            or not math.isfinite(float(self.inference_timeout_seconds))
            or float(self.inference_timeout_seconds) <= 0
        ):
            raise ValueError("inference_timeout_seconds must be positive and finite.")
        if (
            isinstance(self.worker_startup_timeout_seconds, bool)
            or not isinstance(self.worker_startup_timeout_seconds, (int, float))
            or not math.isfinite(float(self.worker_startup_timeout_seconds))
            or float(self.worker_startup_timeout_seconds) <= 0
        ):
            raise ValueError("worker_startup_timeout_seconds must be positive and finite.")
        _, api_problems = validate_secret(
            self.api_token,
            name="api_token",
            required=False,
        )
        _, admin_problems = validate_secret(
            self.admin_token,
            name="admin_token",
            required=False,
        )
        fingerprint_key = resolve_fingerprint_key(self.policy)
        fingerprint_value = (
            None if fingerprint_key is None else fingerprint_key.decode("utf-8")
        )
        secret_problems = (
            *api_problems,
            *admin_problems,
            *validate_distinct_secrets(
                {
                    "api_token": self.api_token,
                    "admin_token": self.admin_token,
                    "fingerprint_key": fingerprint_value,
                }
            ),
        )
        if secret_problems:
            raise ValueError(" ".join(secret_problems))
        if self.target_profile is not None and not self.target_profile.strip():
            raise ValueError("target_profile cannot be empty when provided.")
        self.input_modalities = normalize_input_modalities(self.input_modalities)
        self.traffic_mode = validate_traffic_mode(self.traffic_mode)
        self._artifact = self.artifact_snapshot or load_detector_definition(
            self.artifact_path, self.detector_heads
        )
        self._capacity = BoundedSemaphore(1)
        self._start_worker(initial=True)

    @property
    def ready(self) -> bool:
        _, _, process_alive = self._process_snapshot()
        return bool(self._readiness.get("ok")) and process_alive and not self._closing

    @property
    def startup_warmup(self) -> dict[str, object] | None:
        return self._startup_warmup

    def readiness_report(self) -> dict[str, object]:
        if not self.ready:
            self._ensure_worker_restart()
        process, process_pid, process_alive = self._process_snapshot()
        readiness = dict(self._readiness)
        worker_ready = (
            bool(readiness.get("ok")) and process_alive and not self._closing
        )
        report = readiness
        report["target_profile"] = self.target_profile
        report["capabilities"] = {
            "input_modalities": list(self.input_modalities or REQUEST_MODALITIES),
            "traffic_modes": ["shadow", "review", "enforce"],
            "runtime_traffic_mode_control": self.admin_token is not None,
        }
        report["worker"] = {
            "mode": "process",
            "ready": worker_ready,
            "generation": self._worker_generation,
            "restarts": self._worker_restarts,
            "pid": process_pid,
            "alive": process_alive,
            "inference_timeout_seconds": self.inference_timeout_seconds,
            "last_error": self._last_worker_error,
        }
        report["ok"] = worker_ready
        report["traffic_mode"] = self.current_traffic_mode()
        report["audit_logging"] = self.audit_logger is not None
        if self.audit_logger is not None:
            report.update(self.audit_logger.readiness_context())
        else:
            report["evidence_session_id"] = None
            report["audit_path"] = None
            report["deployment_config_sha256"] = None
            report.setdefault("detector_sha256", None)
        return report

    def _process_snapshot(self) -> tuple[Any, int | None, bool]:
        process = self._process
        if process is None:
            return None, None, False
        try:
            pid = process.pid
        except (AttributeError, AssertionError, OSError, ValueError):
            pid = None
        try:
            alive = bool(process.is_alive())
        except (AssertionError, OSError, ValueError):
            alive = False
        return process, pid, alive

    def authorize(self, authorization: str | None) -> bool:
        if self.api_token is None:
            return True
        if authorization is None or not authorization.startswith("Bearer "):
            return False
        return secret_matches(authorization[7:], self.api_token)

    def authorize_admin(self, authorization: str | None) -> bool:
        if self.admin_token is None:
            return False
        if authorization is None or not authorization.startswith("Bearer "):
            return False
        return secret_matches(authorization[7:], self.admin_token)

    def current_traffic_mode(self) -> str:
        with self._traffic_mode_lock:
            return self.traffic_mode

    def set_traffic_mode(self, traffic_mode: str) -> str:
        return self.change_traffic_mode(traffic_mode)[1]

    def change_traffic_mode(self, traffic_mode: str) -> tuple[str, str]:
        mode = validate_traffic_mode(traffic_mode)
        with self._traffic_mode_lock:
            previous = self.traffic_mode
            self.traffic_mode = mode
            return previous, self.traffic_mode

    def evaluate(self, payload: dict[str, Any]) -> dict[str, object]:
        if not self._capacity.acquire(blocking=False):
            raise ServiceBusyError("Guardrail inference capacity is currently full.")
        try:
            traffic_mode = self.current_traffic_mode()
            if not self.ready:
                self._ensure_worker_restart()
                return apply_traffic_mode(
                    self._fail_safe_response(
                        payload,
                        InferenceWorkerError(
                            "Inference worker is unavailable or restarting."
                        ),
                    ),
                    traffic_mode,
                )
            request_id = str(uuid.uuid4())
            with self._request_lock:
                generation = self._worker_generation
                request_queue = self._request_queue
                response_queue = self._response_queue
                if request_queue is None or response_queue is None:
                    error = InferenceWorkerError(
                        "Inference worker queues are unavailable."
                    )
                    self._invalidate_worker(
                        str(error), expected_generation=generation
                    )
                    return apply_traffic_mode(
                        self._fail_safe_response(payload, error),
                        traffic_mode,
                    )
                try:
                    request_queue.put(
                        {"kind": "evaluate", "id": request_id, "payload": payload},
                        block=False,
                    )
                except Full:
                    raise ServiceBusyError(
                        "Guardrail inference worker queue is currently full."
                    ) from None
                except (EOFError, OSError, ValueError) as exc:
                    error = InferenceWorkerError(
                        f"Inference worker request queue failed: {type(exc).__name__}."
                    )
                    self._invalidate_worker(
                        str(error), expected_generation=generation
                    )
                    return apply_traffic_mode(
                        self._fail_safe_response(payload, error),
                        traffic_mode,
                    )
                try:
                    message = response_queue.get(
                        timeout=self.inference_timeout_seconds
                    )
                except Empty:
                    error = InferenceTimeoutError(
                        "Guardrail inference exceeded the configured wall-clock timeout."
                    )
                    self._invalidate_worker(
                        str(error), expected_generation=generation
                    )
                    return apply_traffic_mode(
                        self._fail_safe_response(payload, error),
                        traffic_mode,
                    )
                except (EOFError, OSError, ValueError) as exc:
                    error = InferenceWorkerError(
                        f"Inference worker response queue failed: {type(exc).__name__}."
                    )
                    self._invalidate_worker(
                        str(error), expected_generation=generation
                    )
                    return apply_traffic_mode(
                        self._fail_safe_response(payload, error),
                        traffic_mode,
                    )
            if not isinstance(message, dict) or message.get("id") != request_id:
                error = InferenceWorkerError("Inference worker returned an invalid response.")
                self._invalidate_worker(
                    str(error), expected_generation=generation
                )
                return apply_traffic_mode(
                    self._fail_safe_response(payload, error),
                    traffic_mode,
                )
            if message.get("kind") == "result":
                result = message.get("payload")
                if not isinstance(result, dict):
                    error = InferenceWorkerError(
                        "Inference worker returned a non-object result."
                    )
                    self._invalidate_worker(
                        str(error), expected_generation=generation
                    )
                    return apply_traffic_mode(
                        self._fail_safe_response(payload, error),
                        traffic_mode,
                    )
                return apply_traffic_mode(result, traffic_mode)
            error_type = str(message.get("error_type", "InferenceWorkerError"))
            error_message = str(message.get("message", "Inference worker failed."))
            if error_type == "ValueError":
                raise ValueError(error_message)
            error = InferenceWorkerError(f"{error_type}: {error_message}")
            self._invalidate_worker(str(error), expected_generation=generation)
            return apply_traffic_mode(
                self._fail_safe_response(payload, error),
                traffic_mode,
            )
        finally:
            self._capacity.release()

    def close(self) -> None:
        self._closing = True
        with self._request_lock:
            with self._lifecycle_lock:
                stopped = self._stop_worker()
                self._readiness = {
                    "ok": False,
                    "checks": [],
                    "reason": (
                        "service_closed"
                        if stopped
                        else "worker_shutdown_incomplete"
                    ),
                }

    def _start_worker(self, *, initial: bool) -> None:
        with self._request_lock:
            with self._lifecycle_lock:
                if self._closing:
                    return
                self._readiness = {
                    "ok": False,
                    "checks": [],
                    "reason": "worker_starting",
                }
                self._startup_warmup = None
                if not self._stop_worker():
                    raise InferenceWorkerError(
                        "The previous inference worker could not be stopped; "
                        "a replacement was not started."
                    )
                try:
                    context = multiprocessing.get_context("spawn")
                    self._request_queue = context.Queue(maxsize=1)
                    self._response_queue = context.Queue(maxsize=2)
                    self._process = context.Process(
                        target=_inference_worker_main,
                        args=(
                            self._request_queue,
                            self._response_queue,
                            None if self.artifact_path is None else str(self.artifact_path),
                            self.detector_heads,
                            self.provider_spec,
                            dict(self.provider_options),
                            self.policy,
                            self.limits,
                            self.warmup_payload,
                            self.target_profile,
                            self.input_modalities,
                        ),
                        name="aegis-inference-worker",
                        daemon=True,
                    )
                    self._process.start()
                    self._worker_generation += 1
                    if not initial:
                        self._worker_restarts += 1
                    deadline = time.monotonic() + self.worker_startup_timeout_seconds
                    message = None
                    while message is None and not self._closing:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise InferenceWorkerError(
                                "Inference worker did not become ready within the "
                                "startup timeout."
                            )
                        try:
                            message = self._response_queue.get(
                                timeout=min(0.25, remaining)
                            )
                        except Empty:
                            if (
                                self._process is not None
                                and not self._process.is_alive()
                            ):
                                raise InferenceWorkerError(
                                    "Inference worker exited before reporting readiness."
                                )
                    if self._closing:
                        self._stop_worker()
                        return
                    if not isinstance(message, dict) or message.get("kind") != "ready":
                        detail = (
                            str(message.get("message"))
                            if isinstance(message, dict)
                            else "invalid startup response"
                        )
                        raise InferenceWorkerError(
                            f"Inference worker failed during startup: {detail}"
                        )
                    readiness = message.get("readiness")
                    if not isinstance(readiness, dict) or not readiness.get("ok"):
                        raise InferenceWorkerError(
                            "Inference worker reported failed provider readiness."
                        )
                    parent_artifact = getattr(self, "_artifact", None)
                    if isinstance(parent_artifact, OrDetector):
                        worker_identity = readiness.get("detector_identity_sha256")
                        if worker_identity != parent_artifact.identity_sha256:
                            raise InferenceWorkerError(
                                "Inference worker detector identity differs from the "
                                "parent's validated detector snapshot."
                            )
                    elif isinstance(parent_artifact, DetectorArtifact):
                        parent_digest = getattr(
                            parent_artifact, "artifact_sha256", None
                        )
                        if (
                            parent_digest is not None
                            and readiness.get("detector_sha256") != parent_digest
                        ):
                            raise InferenceWorkerError(
                                "Inference worker detector identity differs from the "
                                "parent's validated detector snapshot."
                            )
                    self._readiness = readiness
                    warmup = message.get("warmup")
                    self._startup_warmup = (
                        warmup if isinstance(warmup, dict) else None
                    )
                    self._last_worker_error = None
                except BaseException:
                    self._stop_worker()
                    raise

    def _invalidate_worker(
        self,
        detail: str,
        *,
        expected_generation: int | None = None,
    ) -> None:
        with self._request_lock:
            with self._lifecycle_lock:
                if (
                    expected_generation is not None
                    and expected_generation != self._worker_generation
                ):
                    return
                self._last_worker_error = detail
                self._readiness = {"ok": False, "checks": [], "reason": detail}
                stopped = self._stop_worker()
                if not stopped:
                    self._last_worker_error = (
                        "The previous inference worker could not be stopped; "
                        "a replacement was not started."
                    )
                    self._readiness = {
                        "ok": False,
                        "checks": [],
                        "reason": "worker_shutdown_incomplete",
                    }
        if stopped:
            self._ensure_worker_restart()

    def _ensure_worker_restart(self) -> None:
        with self._restart_lock:
            if self._closing or self.ready:
                return
            if self._restart_thread is not None and self._restart_thread.is_alive():
                return
            self._restart_thread = Thread(
                target=self._restart_worker,
                name="aegis-worker-restart",
                daemon=True,
            )
            self._restart_thread.start()

    def _restart_worker(self) -> None:
        delay = _WORKER_RESTART_INITIAL_BACKOFF_SECONDS
        while not self._closing:
            try:
                self._start_worker(initial=False)
                return
            except Exception as exc:
                self._last_worker_error = f"{type(exc).__name__}: {exc}"
                self._readiness = {
                    "ok": False,
                    "checks": [],
                    "reason": self._last_worker_error,
                }
            deadline = time.monotonic() + delay
            while not self._closing:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.25, remaining))
            delay = min(delay * 2.0, _WORKER_RESTART_MAX_BACKOFF_SECONDS)

    def _stop_worker(self) -> bool:
        """Stop the current worker without ever forgetting a live process."""

        process = self._process
        stopped = True
        if process is not None:
            alive = _process_may_be_alive(process)
            if alive:
                try:
                    process.terminate()
                except (OSError, ValueError):
                    pass
                try:
                    process.join(timeout=5.0)
                except (AssertionError, OSError, ValueError):
                    pass
                alive = _process_may_be_alive(process)
            if alive:
                try:
                    process.kill()
                except (AttributeError, OSError, ValueError):
                    pass
                try:
                    process.join(timeout=5.0)
                except (AssertionError, OSError, ValueError):
                    pass
                alive = _process_may_be_alive(process)
            if alive:
                # Retain the immutable process handle. A later restart attempt may
                # retry termination, but it cannot allocate another model worker
                # while this process is still live or its state is uncertain.
                self._process = process
                stopped = False
            else:
                self._process = None
        for queue in (self._request_queue, self._response_queue):
            if queue is not None:
                # Once the worker has been terminated there may be no reader for a
                # partially flushed multi-megabyte request. Waiting for the queue's
                # feeder thread can then block forever and defeat the inference
                # timeout. These queues are being abandoned, so prevent an implicit
                # feeder join and close their local handles without draining them.
                try:
                    queue.cancel_join_thread()
                except (AttributeError, OSError, ValueError):
                    pass
                try:
                    queue.close()
                except (AttributeError, OSError, ValueError):
                    pass
        self._request_queue = None
        self._response_queue = None
        return stopped

    def _fail_safe_response(
        self,
        payload: dict[str, Any],
        error: Exception,
    ) -> dict[str, object]:
        provider = _FailingProvider(self._artifact, error)
        return evaluate_request_payload(
            self._artifact,
            provider,
            payload,
            policy=self.policy,
            limits=self.limits,
            input_modalities=self.input_modalities,
            target_profile=self.target_profile,
        )


class _FailingProvider:
    def __init__(self, artifact: DetectorArtifact | OrDetector, error: Exception) -> None:
        self.model_family = artifact.model_family
        self.model_id = artifact.model_id
        self.model_revision = artifact.model_revision
        self.tokenizer_revision = artifact.tokenizer_revision
        self.preprocessing_sha256 = artifact.preprocessing_sha256
        self.layer = artifact.layer
        self.pooling = artifact.pooling
        self.feature_dim = artifact.feature_dim
        self._error = error

    def embed(self, request: GuardrailRequest) -> np.ndarray:
        raise self._error


def _process_may_be_alive(process: Any) -> bool:
    """Treat an unreadable process state as live so replacement fails closed."""

    try:
        return bool(process.is_alive())
    except (AssertionError, OSError, ValueError):
        return True


def _inference_worker_main(
    request_queue: Any,
    response_queue: Any,
    artifact_path: str | None,
    detector_heads: tuple[DetectorHeadConfig, ...],
    provider_spec: str,
    provider_options: dict[str, object],
    policy: GuardrailPolicy,
    limits: RequestLimits,
    warmup_payload: dict[str, Any] | None,
    target_profile: str | None,
    input_modalities: tuple[str, ...] | None,
) -> None:
    try:
        artifact = load_detector_definition(artifact_path, detector_heads)
        provider = load_provider(provider_spec, provider_options)
        service = GuardrailHTTPService(
            artifact=artifact,
            provider=provider,
            policy=policy,
            limits=limits,
            max_concurrent_requests=1,
            target_profile=target_profile,
            input_modalities=input_modalities,
        )
        warmup = None
        if warmup_payload is not None:
            warmup = service.evaluate(warmup_payload)
            if any(
                decision.get("verdict") == "guardrail_error"
                for decision in warmup.get("decisions", [])
            ):
                raise RuntimeError("Warmup request produced a guardrail_error decision.")
        response_queue.put(
            {
                "kind": "ready",
                "readiness": service.readiness_report(),
                "warmup": warmup,
            }
        )
        while True:
            message = request_queue.get()
            if message is None:
                return
            request_id = message.get("id") if isinstance(message, dict) else None
            try:
                if not isinstance(message, dict) or message.get("kind") != "evaluate":
                    raise ValueError("Inference worker received an invalid request.")
                payload = message.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("Inference request payload must be an object.")
                result = service.evaluate(payload)
                response_queue.put(
                    {"kind": "result", "id": request_id, "payload": result}
                )
            except Exception as exc:
                response_queue.put(
                    {
                        "kind": "error",
                        "id": request_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
    except BaseException as exc:
        response_queue.put(
            {
                "kind": "startup_error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
