from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import re
import socket
from statistics import mean
from threading import BoundedSemaphore, Condition, Lock, Timer
import time
from typing import Any
import uuid

from AEGIS.audit import (
    PrivacySafeAuditLogger,
    apply_traffic_mode,
    validate_traffic_mode,
)
from AEGIS.detector_artifact import DetectorArtifact
from AEGIS.detector_set import OrDetector
from AEGIS.guardrail import EmbeddingProvider, GuardrailPolicy, resolve_fingerprint_key
from AEGIS.http_framing import RequestFramingError, parse_request_content_length
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
from AEGIS.strict_json import strict_json_loads


_SAFE_TRACE_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")


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
        if (
            isinstance(self.recent_latency_limit, bool)
            or not isinstance(self.recent_latency_limit, int)
            or self.recent_latency_limit <= 0
        ):
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
    artifact: DetectorArtifact | OrDetector
    provider: EmbeddingProvider
    policy: GuardrailPolicy
    limits: RequestLimits = field(default_factory=RequestLimits)
    api_token: str | None = field(default=None, repr=False)
    admin_token: str | None = field(default=None, repr=False)
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
        if (
            isinstance(self.max_body_bytes, bool)
            or not isinstance(self.max_body_bytes, int)
            or self.max_body_bytes <= 0
        ):
            raise ValueError("max_body_bytes must be positive.")
        if (
            isinstance(self.max_concurrent_requests, bool)
            or not isinstance(self.max_concurrent_requests, int)
            or self.max_concurrent_requests <= 0
        ):
            raise ValueError("max_concurrent_requests must be positive.")
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
            "runtime_traffic_mode_control": self.admin_token is not None,
        }
        report["traffic_mode"] = self.current_traffic_mode()
        if isinstance(self.artifact, OrDetector):
            detector_status = self.artifact.status()
            report["detector_mode"] = detector_status["mode"]
            report["detector_identity_sha256"] = detector_status["identity_sha256"]
            report["detector_heads"] = detector_status["heads"]
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
        if not isinstance(self.artifact, OrDetector):
            snapshot_digest = getattr(self.artifact, "artifact_sha256", None)
            if snapshot_digest is not None:
                report["detector_sha256"] = snapshot_digest
        cache = getattr(self.provider, "_feature_cache", None)
        if cache is not None and hasattr(cache, "stats"):
            report["request_cache"] = cache.stats().__dict__
        return report

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
    artifact: DetectorArtifact | OrDetector,
    provider: EmbeddingProvider,
    *,
    require_matching_provenance: bool = True,
) -> dict[str, object]:
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    for name in (
        "model_family",
        "model_id",
        "layer",
        "pooling",
        "feature_dim",
        "embed",
    ):
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
        names = ["model_family", "model_id", "layer", "pooling"]
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
        "layer": getattr(provider, "layer", None),
        "pooling": getattr(provider, "pooling", None),
        "feature_dim": provider_dim,
    }


def public_readiness_report(report: dict[str, object]) -> dict[str, object]:
    """Return the unauthenticated health contract without diagnostics."""

    target_profile = report.get("target_profile")
    if not isinstance(target_profile, str) or not target_profile:
        target_profile = None
    traffic_mode = report.get("traffic_mode")
    if traffic_mode not in {"shadow", "review", "enforce"}:
        traffic_mode = None

    raw_capabilities = report.get("capabilities")
    capabilities: dict[str, object] = {}
    if isinstance(raw_capabilities, dict):
        raw_modalities = raw_capabilities.get("input_modalities")
        if isinstance(raw_modalities, (list, tuple)):
            capabilities["input_modalities"] = [
                value
                for value in raw_modalities
                if isinstance(value, str) and value in REQUEST_MODALITIES
            ]
        raw_modes = raw_capabilities.get("traffic_modes")
        if isinstance(raw_modes, (list, tuple)):
            capabilities["traffic_modes"] = [
                value
                for value in raw_modes
                if isinstance(value, str)
                and value in {"shadow", "review", "enforce"}
            ]
        capabilities["runtime_traffic_mode_control"] = (
            raw_capabilities.get("runtime_traffic_mode_control") is True
        )

    ready = report.get("ok") is True
    public = {
        "ok": ready,
        "target_profile": target_profile,
        "traffic_mode": traffic_mode,
        "capabilities": capabilities,
    }
    if not ready:
        public["reason"] = "service_not_ready"
    return public


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Thread-per-connection server with connection and absolute read bounds."""

    # Request threads cannot be allowed to keep process shutdown unbounded. The
    # explicit activity counter below provides a bounded drain for valid work;
    # daemon threads are then a last-resort exit path for a defective in-process
    # provider that ignores every configured inference bound.
    daemon_threads = True

    def __init__(
        self,
        server_address,
        request_handler_class,
        *,
        max_http_connections: int,
        request_read_timeout_seconds: float,
    ) -> None:
        if (
            isinstance(max_http_connections, bool)
            or not isinstance(max_http_connections, int)
            or max_http_connections <= 0
        ):
            raise ValueError("max_http_connections must be positive.")
        if (
            isinstance(request_read_timeout_seconds, bool)
            or not isinstance(request_read_timeout_seconds, (int, float))
            or not math.isfinite(float(request_read_timeout_seconds))
            or float(request_read_timeout_seconds) <= 0
        ):
            raise ValueError("request_read_timeout_seconds must be positive and finite.")
        self.max_http_connections = int(max_http_connections)
        self.request_read_timeout_seconds = float(request_read_timeout_seconds)
        self._connection_capacity = BoundedSemaphore(self.max_http_connections)
        self._request_activity = Condition(Lock())
        self._active_request_threads = 0
        self._read_deadline_lock = Lock()
        self._read_deadline_timers: dict[socket.socket, Timer] = {}
        self._expired_read_deadlines: set[socket.socket] = set()
        super().__init__(server_address, request_handler_class)

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(self.request_read_timeout_seconds)
        return request, client_address

    def process_request(self, request, client_address) -> None:
        if not self._connection_capacity.acquire(blocking=False):
            self.shutdown_request(request)
            return
        self._request_started()
        try:
            self._arm_request_read_deadline(request)
            super().process_request(request, client_address)
        except BaseException:
            self.finish_request_read(request)
            self._connection_capacity.release()
            self._request_finished()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.finish_request_read(request)
            self._connection_capacity.release()
            self._request_finished()

    def wait_for_request_threads(self, timeout_seconds: float) -> bool:
        """Wait for accepted handlers under one finite shutdown deadline."""

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite.")
        deadline = time.monotonic() + float(timeout_seconds)
        with self._request_activity:
            while self._active_request_threads:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._request_activity.wait(timeout=remaining)
        return True

    def _request_started(self) -> None:
        with self._request_activity:
            self._active_request_threads += 1

    def _request_finished(self) -> None:
        with self._request_activity:
            self._active_request_threads -= 1
            if self._active_request_threads == 0:
                self._request_activity.notify_all()

    def finish_request_read(self, request: socket.socket) -> bool:
        """Cancel the absolute read deadline and report whether it expired."""

        with self._read_deadline_lock:
            timer = self._read_deadline_timers.pop(request, None)
            expired = request in self._expired_read_deadlines
            self._expired_read_deadlines.discard(request)
        if timer is not None:
            timer.cancel()
        return expired

    def request_read_deadline_expired(self, request: socket.socket) -> bool:
        with self._read_deadline_lock:
            return request in self._expired_read_deadlines

    def handle_error(self, request, client_address) -> None:
        if self.request_read_deadline_expired(request):
            return
        super().handle_error(request, client_address)

    def _arm_request_read_deadline(self, request: socket.socket) -> None:
        timer = Timer(
            self.request_read_timeout_seconds,
            self._expire_request_read,
            args=(request,),
        )
        timer.daemon = True
        with self._read_deadline_lock:
            self._read_deadline_timers[request] = timer
        try:
            timer.start()
        except BaseException:
            with self._read_deadline_lock:
                current = self._read_deadline_timers.get(request)
                if current is timer:
                    self._read_deadline_timers.pop(request, None)
            timer.cancel()
            raise

    def _expire_request_read(self, request: socket.socket) -> None:
        with self._read_deadline_lock:
            timer = self._read_deadline_timers.pop(request, None)
            if timer is None:
                return
            self._expired_read_deadlines.add(request)
        try:
            request.shutdown(socket.SHUT_RD)
        except OSError:
            pass


class BoundedIPv6ThreadingHTTPServer(BoundedThreadingHTTPServer):
    address_family = socket.AF_INET6


def create_guardrail_http_server(
    host: str,
    port: int,
    service: Any,
    *,
    max_http_connections: int = 32,
    request_read_timeout_seconds: float = 15.0,
) -> BoundedThreadingHTTPServer:
    require_token_for_bind(host, getattr(service, "api_token", None))
    handler = _handler_class(service)
    try:
        address = ipaddress.ip_address(str(host).strip())
    except ValueError:
        address = None
    server_class = (
        BoundedIPv6ThreadingHTTPServer
        if isinstance(address, ipaddress.IPv6Address)
        else BoundedThreadingHTTPServer
    )
    return server_class(
        (host, port),
        handler,
        max_http_connections=max_http_connections,
        request_read_timeout_seconds=request_read_timeout_seconds,
    )


def require_token_for_bind(host: str, api_token: str | None) -> None:
    normalized = host.strip().lower()
    loopback = normalized in {"127.0.0.1", "localhost", "::1"}
    if not loopback:
        _, problems = validate_secret(
            api_token,
            name="Non-loopback API token",
            required=True,
        )
        if problems:
            raise ValueError(" ".join(problems))


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
                self._respond(
                    public_readiness_report(report),
                    status=status,
                    started=start,
                )
                return
            if self.path == "/v1/status":
                if not self._authorized(start):
                    return
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
                "Use GET /livez, /readyz, /v1/status, /metrics or POST /v1/guard.",
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
            length = self._required_content_length(start)
            if length is None:
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
                encoded_body = self.rfile.read(length)
            except (TimeoutError, socket.timeout, OSError):
                self._error(
                    HTTPStatus.REQUEST_TIMEOUT,
                    "request_timeout",
                    "The request body was not received within the configured timeout.",
                    start,
                )
                return
            if self._finish_request_read() or len(encoded_body) != length:
                self._error(
                    HTTPStatus.REQUEST_TIMEOUT,
                    "request_timeout",
                    "The request body was not received within the configured timeout.",
                    start,
                )
                return
            try:
                payload = strict_json_loads(encoded_body.decode("utf-8"))
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
            length = self._required_content_length(start)
            if length is None:
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
                encoded_body = self.rfile.read(length)
            except (TimeoutError, socket.timeout, OSError):
                self._error(
                    HTTPStatus.REQUEST_TIMEOUT,
                    "request_timeout",
                    "The request body was not received within the configured timeout.",
                    start,
                )
                return
            if self._finish_request_read() or len(encoded_body) != length:
                self._error(
                    HTTPStatus.REQUEST_TIMEOUT,
                    "request_timeout",
                    "The request body was not received within the configured timeout.",
                    start,
                )
                return
            try:
                payload = strict_json_loads(encoded_body.decode("utf-8"))
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
            if getattr(service, "admin_token", None) is None:
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "runtime_control_disabled",
                    "Runtime controls require a distinct configured admin bearer token.",
                    started,
                )
                return False
            if service.authorize_admin(self.headers.get("Authorization")):
                return True
            self._error(
                HTTPStatus.UNAUTHORIZED,
                "unauthorized",
                "A valid admin bearer token is required.",
                started,
                extra_headers={"WWW-Authenticate": "Bearer"},
            )
            return False

        def _required_content_length(self, started: float) -> int | None:
            try:
                return parse_request_content_length(
                    self.headers,
                    body_required=True,
                )
            except RequestFramingError as exc:
                self.close_connection = True
                if exc.reason in {
                    "missing_content_length",
                    "invalid_content_length",
                    "body_required",
                }:
                    self._error(
                        HTTPStatus.LENGTH_REQUIRED,
                        "content_length_required",
                        "A positive Content-Length is required.",
                        started,
                    )
                else:
                    self._error(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_request",
                        str(exc),
                        started,
                    )
                return None

        def _finish_request_read(self) -> bool:
            finish = getattr(self.server, "finish_request_read", None)
            return bool(finish(self.connection)) if callable(finish) else False

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
            if _SAFE_TRACE_ID.fullmatch(incoming) is not None:
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
            self._finish_request_read()
            body = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
            try:
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
            except (OSError, socket.timeout):
                self.close_connection = True
                service.metrics.record(
                    status=int(status),
                    duration_seconds=time.perf_counter() - started,
                    actions=actions,
                    guardrail_errors=guardrail_errors,
                )
                return
            service.metrics.record(
                status=int(status),
                duration_seconds=time.perf_counter() - started,
                actions=actions,
                guardrail_errors=guardrail_errors,
            )
            try:
                self.wfile.write(body)
            except (OSError, socket.timeout):
                self.close_connection = True

    return Handler


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    index = int(round((len(values) - 1) * probability))
    return float(values[index])
