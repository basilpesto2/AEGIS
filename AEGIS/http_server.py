from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
from statistics import mean
from threading import BoundedSemaphore, Lock
import time
from typing import Any
import uuid

from AEGIS.audit import (
    PrivacySafeAuditLogger,
    apply_traffic_mode,
    validate_traffic_mode,
)
from AEGIS.detector_artifact import DetectorArtifact
from AEGIS.guardrail import EmbeddingProvider, GuardrailPolicy, resolve_fingerprint_key
from AEGIS.service import (
    REQUEST_MODALITIES,
    RequestLimits,
    evaluate_request_payload,
    normalize_input_modalities,
)


@dataclass
class ServiceMetrics:
    recent_latency_limit: int = 1_000
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _requests: int = field(default=0, init=False)
    _guardrail_errors: int = field(default=0, init=False)
    _status_counts: Counter[int] = field(default_factory=Counter, init=False, repr=False)
    _action_counts: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    _latencies: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.recent_latency_limit <= 0:
            raise ValueError("recent_latency_limit must be positive.")
        self._latencies = deque(maxlen=self.recent_latency_limit)

    def record(
        self,
        *,
        status: int,
        duration_seconds: float,
        actions: dict[str, int] | None = None,
        guardrail_errors: int = 0,
    ) -> None:
        with self._lock:
            self._requests += 1
            self._status_counts[int(status)] += 1
            self._guardrail_errors += int(guardrail_errors)
            self._latencies.append(float(duration_seconds))
            if actions:
                self._action_counts.update(actions)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            latencies = sorted(self._latencies)
            return {
                "requests_total": self._requests,
                "http_status_counts": {
                    str(key): value for key, value in sorted(self._status_counts.items())
                },
                "action_counts": dict(sorted(self._action_counts.items())),
                "guardrail_errors_total": self._guardrail_errors,
                "latency_seconds": {
                    "n": len(latencies),
                    "mean": mean(latencies) if latencies else None,
                    "p50": _quantile(latencies, 0.50),
                    "p95": _quantile(latencies, 0.95),
                    "p99": _quantile(latencies, 0.99),
                    "max": max(latencies) if latencies else None,
                },
            }


@dataclass
class GuardrailHTTPService:
    artifact: DetectorArtifact
    provider: EmbeddingProvider
    policy: GuardrailPolicy
    limits: RequestLimits = field(default_factory=RequestLimits)
    api_token: str | None = None
    max_body_bytes: int = 12 * 1024 * 1024
    max_concurrent_requests: int = 1
    traffic_mode: str = "enforce"
    audit_logger: PrivacySafeAuditLogger | None = None
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)
    target_profile: str | None = None
    input_modalities: tuple[str, ...] | None = None
    _capacity: BoundedSemaphore = field(init=False, repr=False)
    _traffic_mode_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _readiness: dict[str, object] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive.")
        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive.")
        if self.api_token is not None and not self.api_token:
            raise ValueError("api_token cannot be empty when provided.")
        if self.target_profile is not None and not self.target_profile.strip():
            raise ValueError("target_profile cannot be empty when provided.")
        self.input_modalities = normalize_input_modalities(self.input_modalities)
        resolve_fingerprint_key(self.policy)
        self.traffic_mode = validate_traffic_mode(self.traffic_mode)
        self._capacity = BoundedSemaphore(self.max_concurrent_requests)
        self._readiness = provider_readiness_report(
            self.artifact,
            self.provider,
            require_matching_provenance=self.policy.require_matching_provenance,
        )

    @property
    def ready(self) -> bool:
        return bool(self._readiness.get("ok"))

    def readiness_report(self) -> dict[str, object]:
        report = dict(self._readiness)
        report["target_profile"] = self.target_profile
        report["capabilities"] = {
            "input_modalities": list(self.input_modalities or REQUEST_MODALITIES),
            "traffic_modes": ["shadow", "review", "enforce"],
            "runtime_traffic_mode_control": self.api_token is not None,
        }
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
        cache = getattr(self.provider, "_feature_cache", None)
        if cache is not None and hasattr(cache, "stats"):
            report["request_cache"] = cache.stats().__dict__
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
            response = evaluate_request_payload(
                self.artifact,
                self.provider,
                payload,
                policy=self.policy,
                limits=self.limits,
                input_modalities=self.input_modalities,
                target_profile=self.target_profile,
            )
            return apply_traffic_mode(response, traffic_mode)
        finally:
            self._capacity.release()


class ServiceBusyError(RuntimeError):
    pass


def provider_readiness_report(
    artifact: DetectorArtifact,
    provider: EmbeddingProvider,
    *,
    require_matching_provenance: bool = True,
) -> dict[str, object]:
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    for name in ("model_family", "model_id", "pooling", "feature_dim", "embed"):
        check(
            f"provider_has_{name}",
            hasattr(provider, name),
            f"value={getattr(provider, name, None)!r}",
        )
    try:
        provider_dim = int(getattr(provider, "feature_dim"))
    except (AttributeError, TypeError, ValueError):
        provider_dim = -1
    check(
        "feature_dim_matches_detector",
        provider_dim == artifact.feature_dim,
        f"provider={provider_dim}, detector={artifact.feature_dim}",
    )
    if require_matching_provenance:
        names = ["model_family", "model_id", "pooling"]
        names.extend(
            name
            for name in ("model_revision", "tokenizer_revision", "preprocessing_sha256")
            if getattr(artifact, name)
        )
        for name in names:
            check(
                f"{name}_matches_detector",
                getattr(provider, name, None) == getattr(artifact, name),
                (
                    f"provider={getattr(provider, name, None)!r}, "
                    f"detector={getattr(artifact, name)!r}"
                ),
            )
    return {
        "ok": all(bool(item["ok"]) for item in checks),
        "checks": checks,
        "model_family": getattr(provider, "model_family", None),
        "model_id": getattr(provider, "model_id", None),
        "model_revision": getattr(provider, "model_revision", None),
        "tokenizer_revision": getattr(provider, "tokenizer_revision", None),
        "preprocessing_sha256": getattr(provider, "preprocessing_sha256", None),
        "pooling": getattr(provider, "pooling", None),
        "feature_dim": provider_dim,
    }


def create_guardrail_http_server(
    host: str,
    port: int,
    service: Any,
) -> ThreadingHTTPServer:
    handler = _handler_class(service)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def require_token_for_bind(host: str, api_token: str | None) -> None:
    normalized = host.strip().lower()
    loopback = normalized in {"127.0.0.1", "localhost", "::1"}
    if not loopback and not api_token:
        raise ValueError("Non-loopback binding requires an API token.")


def _handler_class(service: Any):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AEGIS"
        sys_version = ""

        def do_GET(self) -> None:
            start = time.perf_counter()
            if self.path == "/livez":
                self._respond({"ok": True}, status=HTTPStatus.OK, started=start)
                return
            if self.path in {"/readyz", "/healthz"}:
                report = service.readiness_report()
                status = HTTPStatus.OK if report["ok"] else HTTPStatus.SERVICE_UNAVAILABLE
                self._respond(report, status=status, started=start)
                return
            if self.path == "/metrics":
                if not self._authorized(start):
                    return
                self._respond(service.metrics.snapshot(), status=HTTPStatus.OK, started=start)
                return
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                "Use GET /livez, /readyz, /metrics or POST /v1/guard.",
                start,
            )

        def do_POST(self) -> None:
            start = time.perf_counter()
            if self.path != "/v1/guard":
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "not_found",
                    "Use POST /v1/guard.",
                    start,
                )
                return
            if not self._authorized(start):
                return
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != "application/json":
                self._error(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "unsupported_media_type",
                    "Content-Type must be application/json.",
                    start,
                )
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if length <= 0:
                self._error(
                    HTTPStatus.LENGTH_REQUIRED,
                    "content_length_required",
                    "A positive Content-Length is required.",
                    start,
                )
                return
            if length > service.max_body_bytes:
                self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    f"Request body exceeds {service.max_body_bytes} bytes.",
                    start,
                )
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("JSON request body must be an object.")
                response = service.evaluate(payload)
            except ServiceBusyError as exc:
                self._error(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    "guardrail_busy",
                    str(exc),
                    start,
                )
                return
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_request",
                    str(exc),
                    start,
                )
                return
            except Exception:
                self._error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "The guardrail service encountered an unexpected error.",
                    start,
                )
                return

            trace_id = self._trace_id()
            response["trace_id"] = trace_id
            summary = response.get("summary", {})
            response_traffic_mode = (
                summary.get("traffic_mode")
                if isinstance(summary, dict)
                else service.current_traffic_mode()
            )
            if service.audit_logger is not None:
                try:
                    response["audit_event_ids"] = service.audit_logger.record_response(
                        response,
                        traffic_mode=response_traffic_mode,
                        status=int(HTTPStatus.OK),
                        duration_seconds=time.perf_counter() - start,
                    )
                except (OSError, ValueError):
                    self._error(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "audit_unavailable",
                        "The required privacy-safe audit record could not be written.",
                        start,
                    )
                    return
            action_counts = summary.get("action_counts", {})
            guardrail_errors = sum(
                1
                for decision in response.get("decisions", [])
                if decision.get("verdict") == "guardrail_error"
            )
            self._respond(
                response,
                status=HTTPStatus.OK,
                started=start,
                actions=action_counts if isinstance(action_counts, dict) else None,
                guardrail_errors=guardrail_errors,
                trace_id=trace_id,
            )

        def do_PUT(self) -> None:
            start = time.perf_counter()
            if self.path != "/v1/admin/traffic-mode":
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "not_found",
                    "Use PUT /v1/admin/traffic-mode.",
                    start,
                )
                return
            if not self._admin_authorized(start):
                return
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != "application/json":
                self._error(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "unsupported_media_type",
                    "Content-Type must be application/json.",
                    start,
                )
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if length <= 0:
                self._error(
                    HTTPStatus.LENGTH_REQUIRED,
                    "content_length_required",
                    "A positive Content-Length is required.",
                    start,
                )
                return
            if length > 4096:
                self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    "Traffic-mode request body exceeds 4096 bytes.",
                    start,
                )
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("JSON request body must be an object.")
                if set(payload) != {"traffic_mode"}:
                    raise ValueError(
                        "Traffic-mode request must contain only 'traffic_mode'."
                    )
                requested_mode = payload["traffic_mode"]
                if not isinstance(requested_mode, str):
                    raise ValueError("traffic_mode must be a string.")
                previous_mode, traffic_mode = service.change_traffic_mode(
                    requested_mode
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_request",
                    str(exc),
                    start,
                )
                return
            self._respond(
                {
                    "target_profile": service.target_profile,
                    "previous_traffic_mode": previous_mode,
                    "traffic_mode": traffic_mode,
                    "changed": traffic_mode != previous_mode,
                },
                status=HTTPStatus.OK,
                started=start,
            )

        def log_message(self, format: str, *args) -> None:
            return

        def _authorized(self, started: float) -> bool:
            if service.authorize(self.headers.get("Authorization")):
                return True
            self._error(
                HTTPStatus.UNAUTHORIZED,
                "unauthorized",
                "A valid bearer token is required.",
                started,
                extra_headers={"WWW-Authenticate": "Bearer"},
            )
            return False

        def _admin_authorized(self, started: float) -> bool:
            if getattr(service, "api_token", None) is None:
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "runtime_control_disabled",
                    "Runtime controls require a configured API bearer token.",
                    started,
                )
                return False
            return self._authorized(started)

        def _error(
            self,
            status: HTTPStatus,
            code: str,
            message: str,
            started: float,
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self._respond(
                {
                    "error": {
                        "code": code,
                        "message": message,
                        "trace_id": self._trace_id(),
                    }
                },
                status=status,
                started=started,
                extra_headers=extra_headers,
            )

        def _trace_id(self) -> str:
            incoming = self.headers.get("X-Request-ID", "").strip()
            if incoming and len(incoming) <= 128 and incoming.isascii():
                return incoming
            return str(uuid.uuid4())

        def _respond(
            self,
            payload: dict[str, object],
            *,
            status: HTTPStatus,
            started: float,
            actions: dict[str, int] | None = None,
            guardrail_errors: int = 0,
            trace_id: str | None = None,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            body = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if trace_id:
                self.send_header("X-Request-ID", trace_id)
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            service.metrics.record(
                status=int(status),
                duration_seconds=time.perf_counter() - started,
                actions=actions,
                guardrail_errors=guardrail_errors,
            )
            self.wfile.write(body)

    return Handler


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    index = int(round((len(values) - 1) * probability))
    return float(values[index])
