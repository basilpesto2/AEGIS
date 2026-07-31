from __future__ import annotations

from dataclasses import dataclass, field
import hmac
import multiprocessing
from pathlib import Path
from queue import Empty, Full
from threading import BoundedSemaphore, Lock, Thread
from typing import Any
import time
import uuid

import numpy as np

from AEGIS.audit import (
    PrivacySafeAuditLogger,
    apply_traffic_mode,
    validate_traffic_mode,
)
from AEGIS.detector_artifact import DetectorArtifact, load_detector_artifact
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


class InferenceTimeoutError(TimeoutError):
    pass


class InferenceWorkerError(RuntimeError):
    pass


@dataclass
class ProcessIsolatedGuardrailService:
    artifact_path: Path
    provider_spec: str
    provider_options: dict[str, object]
    policy: GuardrailPolicy
    limits: RequestLimits = field(default_factory=RequestLimits)
    api_token: str | None = None
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
    _artifact: DetectorArtifact = field(init=False, repr=False)
    _capacity: BoundedSemaphore = field(init=False, repr=False)
    _request_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _lifecycle_lock: Lock = field(default_factory=Lock, init=False, repr=False)
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
        if self.max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive.")
        if self.max_concurrent_requests != 1:
            raise ValueError(
                "Process-isolated inference currently requires "
                "server.max_concurrent_requests=1."
            )
        if self.inference_timeout_seconds <= 0:
            raise ValueError("inference_timeout_seconds must be positive.")
        if self.worker_startup_timeout_seconds <= 0:
            raise ValueError("worker_startup_timeout_seconds must be positive.")
        if self.api_token is not None and not self.api_token:
            raise ValueError("api_token cannot be empty when provided.")
        if self.target_profile is not None and not self.target_profile.strip():
            raise ValueError("target_profile cannot be empty when provided.")
        self.input_modalities = normalize_input_modalities(self.input_modalities)
        resolve_fingerprint_key(self.policy)
        self.traffic_mode = validate_traffic_mode(self.traffic_mode)
        self._artifact = load_detector_artifact(self.artifact_path)
        self._capacity = BoundedSemaphore(1)
        self._start_worker(initial=True)

    @property
    def ready(self) -> bool:
        process_alive = self._process is not None and self._process.is_alive()
        return bool(self._readiness.get("ok")) and process_alive and not self._closing

    @property
    def startup_warmup(self) -> dict[str, object] | None:
        return self._startup_warmup

    def readiness_report(self) -> dict[str, object]:
        report = dict(self._readiness)
        report["target_profile"] = self.target_profile
        report["capabilities"] = {
            "input_modalities": list(self.input_modalities or REQUEST_MODALITIES),
            "traffic_modes": ["shadow", "review", "enforce"],
            "runtime_traffic_mode_control": self.api_token is not None,
        }
        report["worker"] = {
            "mode": "process",
            "ready": self.ready,
            "generation": self._worker_generation,
            "restarts": self._worker_restarts,
            "pid": None if self._process is None else self._process.pid,
            "alive": self._process is not None and self._process.is_alive(),
            "inference_timeout_seconds": self.inference_timeout_seconds,
            "last_error": self._last_worker_error,
        }
        report["ok"] = self.ready
        report["traffic_mode"] = self.current_traffic_mode()
        report["audit_logging"] = self.audit_logger is not None
        if self.audit_logger is not None:
            report.update(self.audit_logger.readiness_context())
        else:
            report.update(
                {
                    "evidence_session_id": None,
                    "audit_path": None,
                    "deployment_config_sha256": None,
                    "detector_sha256": None,
                }
            )
        return report

    def authorize(self, authorization: str | None) -> bool:
        if self.api_token is None:
            return True
        if authorization is None or not authorization.startswith("Bearer "):
            return False
        return hmac.compare_digest(authorization[7:], self.api_token)

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
                try:
                    self._request_queue.put(
                        {"kind": "evaluate", "id": request_id, "payload": payload},
                        block=False,
                    )
                except Full:
                    raise ServiceBusyError(
                        "Guardrail inference worker queue is currently full."
                    ) from None
                try:
                    message = self._response_queue.get(
                        timeout=self.inference_timeout_seconds
                    )
                except Empty:
                    error = InferenceTimeoutError(
                        "Guardrail inference exceeded the configured wall-clock timeout."
                    )
                    self._invalidate_worker(str(error))
                    return apply_traffic_mode(
                        self._fail_safe_response(payload, error),
                        traffic_mode,
                    )
            if not isinstance(message, dict) or message.get("id") != request_id:
                error = InferenceWorkerError("Inference worker returned an invalid response.")
                self._invalidate_worker(str(error))
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
                    self._invalidate_worker(str(error))
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
            self._invalidate_worker(str(error))
            return apply_traffic_mode(
                self._fail_safe_response(payload, error),
                traffic_mode,
            )
        finally:
            self._capacity.release()

    def close(self) -> None:
        self._closing = True
        with self._lifecycle_lock:
            self._stop_worker()
            self._readiness = {"ok": False, "checks": [], "reason": "service_closed"}

    def _start_worker(self, *, initial: bool) -> None:
        with self._lifecycle_lock:
            if self._closing:
                return
            self._stop_worker()
            context = multiprocessing.get_context("spawn")
            self._request_queue = context.Queue(maxsize=1)
            self._response_queue = context.Queue(maxsize=2)
            self._process = context.Process(
                target=_inference_worker_main,
                args=(
                    self._request_queue,
                    self._response_queue,
                    str(self.artifact_path),
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
                    self._stop_worker()
                    raise InferenceWorkerError(
                        "Inference worker did not become ready within the startup timeout."
                    )
                try:
                    message = self._response_queue.get(timeout=min(0.25, remaining))
                except Empty:
                    if self._process is not None and not self._process.is_alive():
                        self._stop_worker()
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
                self._stop_worker()
                raise InferenceWorkerError(
                    f"Inference worker failed during startup: {detail}"
                )
            readiness = message.get("readiness")
            if not isinstance(readiness, dict) or not readiness.get("ok"):
                self._stop_worker()
                raise InferenceWorkerError(
                    "Inference worker reported failed provider readiness."
                )
            self._readiness = readiness
            warmup = message.get("warmup")
            self._startup_warmup = warmup if isinstance(warmup, dict) else None
            self._last_worker_error = None

    def _invalidate_worker(self, detail: str) -> None:
        self._last_worker_error = detail
        self._readiness = {"ok": False, "checks": [], "reason": detail}
        self._stop_worker()
        if self._closing or (
            self._restart_thread is not None and self._restart_thread.is_alive()
        ):
            return
        self._restart_thread = Thread(
            target=self._restart_worker,
            name="aegis-worker-restart",
            daemon=True,
        )
        self._restart_thread.start()

    def _restart_worker(self) -> None:
        try:
            self._start_worker(initial=False)
        except Exception as exc:
            self._last_worker_error = f"{type(exc).__name__}: {exc}"
            self._readiness = {
                "ok": False,
                "checks": [],
                "reason": self._last_worker_error,
            }

    def _stop_worker(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
        for queue in (self._request_queue, self._response_queue):
            if queue is not None:
                queue.close()
                queue.join_thread()
        self._request_queue = None
        self._response_queue = None

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
    def __init__(self, artifact: DetectorArtifact, error: Exception) -> None:
        self.model_family = artifact.model_family
        self.model_id = artifact.model_id
        self.model_revision = artifact.model_revision
        self.tokenizer_revision = artifact.tokenizer_revision
        self.preprocessing_sha256 = artifact.preprocessing_sha256
        self.pooling = artifact.pooling
        self.feature_dim = artifact.feature_dim
        self._error = error

    def embed(self, request: GuardrailRequest) -> np.ndarray:
        raise self._error


def _inference_worker_main(
    request_queue: Any,
    response_queue: Any,
    artifact_path: str,
    provider_spec: str,
    provider_options: dict[str, object],
    policy: GuardrailPolicy,
    limits: RequestLimits,
    warmup_payload: dict[str, Any] | None,
    target_profile: str | None,
    input_modalities: tuple[str, ...] | None,
) -> None:
    try:
        artifact = load_detector_artifact(artifact_path)
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
