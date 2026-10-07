from __future__ import annotations

import base64
import binascii
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
import ipaddress
from importlib.resources import files
from io import BytesIO
import json
import math
from pathlib import Path
import re
import secrets
import socket
import sqlite3
import time
from threading import Condition, Lock, Thread
from typing import Any, Callable, Iterator
from urllib.parse import parse_qs, urlsplit
import uuid

from PIL import Image, UnidentifiedImageError

from AEGIS.gui_control import (
    ComposeTargetController,
    GUIControlBusyError,
    GUIControlError,
)
from AEGIS.gui_history import (
    GUIHistoryRetentionError,
    GUIHistoryStore,
    StoredImage,
)
from AEGIS.http_server import (
    BoundedIPv6ThreadingHTTPServer,
    BoundedThreadingHTTPServer,
)
from AEGIS.http_framing import (
    RequestFramingError,
    parse_request_content_length,
)
from AEGIS.http_transport import is_loopback_hostname, request_with_deadline
from AEGIS.secret_validation import (
    MIN_SECRET_CHARACTERS,
    secret_matches,
    validate_distinct_secrets,
)
from AEGIS.strict_json import strict_json_loads


_MAX_GUI_BODY_BYTES = 16 * 1024 * 1024
_MAX_UPSTREAM_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_TEXT_CHARACTERS = 32_768
# A 10 MiB image exceeds the shipped service's 12 MiB request cap after base64
# expansion. Eight MiB leaves room for JSON metadata across all bundled targets.
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_IMAGE_PIXELS = 20_000_000
_ALLOWED_IMAGE_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
_INPUT_MODALITIES = ("text", "image", "image_text")
_TRAFFIC_MODES = ("shadow", "review", "enforce")
_GUI_SESSION_HEADER = "X-AEGIS-GUI-Session"
_STATIC_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/history": ("index.html", "text/html; charset=utf-8"),
    "/settings": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/assets/app.css": ("app.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'none'; connect-src 'self'; "
        "font-src 'self'; form-action 'self'; frame-ancestors 'none'; "
        "img-src 'self' blob: data:; object-src 'none'; script-src 'self'; "
        "style-src 'self'"
    ),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), geolocation=(), microphone=()",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


class GUISubmissionError(ValueError):
    pass


class GUIShuttingDownError(RuntimeError):
    pass


class GUITargetChangingError(RuntimeError):
    pass


class UpstreamGuardrailError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "upstream_error",
        status: int = HTTPStatus.BAD_GATEWAY,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = int(status)


class _HistoryActivityBarrier:
    """Give history clears a deterministic boundary around active evaluations."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._active_evaluations = 0
        self._clearing = False
        self._changing_target = False
        self._draining = False

    @contextmanager
    def evaluation(self) -> Iterator[None]:
        with self._condition:
            while self._clearing and not self._draining:
                self._condition.wait()
            if self._draining:
                raise GUIShuttingDownError("The dashboard is shutting down.")
            if self._changing_target:
                raise GUITargetChangingError("The active detector target is changing.")
            self._active_evaluations += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_evaluations -= 1
                self._condition.notify_all()

    def clear(self, operation: Callable[[], dict[str, int]]) -> dict[str, int]:
        with self._condition:
            while (self._clearing or self._changing_target) and not self._draining:
                self._condition.wait()
            if self._draining:
                raise GUIShuttingDownError("The dashboard is shutting down.")
            self._clearing = True
            while self._active_evaluations:
                self._condition.wait()
        try:
            return operation()
        finally:
            with self._condition:
                self._clearing = False
                self._condition.notify_all()

    @contextmanager
    def target_change(self) -> Iterator[None]:
        """Drain evaluations and reject new ones while a target is replaced."""

        with self._condition:
            if self._draining:
                raise GUIShuttingDownError("The dashboard is shutting down.")
            if self._changing_target:
                raise GUIControlBusyError()
            self._changing_target = True
            while self._active_evaluations or self._clearing:
                self._condition.wait()
                if self._draining:
                    self._changing_target = False
                    self._condition.notify_all()
                    raise GUIShuttingDownError("The dashboard is shutting down.")
        try:
            yield
        finally:
            with self._condition:
                self._changing_target = False
                self._condition.notify_all()

    def wait_until_idle(self) -> None:
        with self._condition:
            while (
                self._active_evaluations
                or self._clearing
                or self._changing_target
            ):
                self._condition.wait()

    def begin_shutdown(self) -> None:
        with self._condition:
            self._draining = True
            self._condition.notify_all()

    def begin_shutdown_and_wait(self) -> None:
        self.begin_shutdown()
        self.wait_until_idle()


class _GUILifecycle:
    """Coordinate one idempotent drain across HTTP and process shutdown paths."""

    def __init__(
        self,
        history_activity: _HistoryActivityBarrier,
        target_controller: ComposeTargetController,
    ) -> None:
        self._history_activity = history_activity
        self._target_controller = target_controller
        self._condition = Condition()
        self._requested = False
        self._waiting = False
        self._drained = False

    @property
    def requested(self) -> bool:
        with self._condition:
            return self._requested

    def begin(self) -> bool:
        """Reject new work immediately and report whether drain was already requested."""

        with self._condition:
            already_requested = self._requested
            if already_requested:
                return True
            self._requested = True
            self._history_activity.begin_shutdown()
            self._target_controller.begin_shutdown()
            self._condition.notify_all()
            return False

    def begin_and_wait(self) -> None:
        self.begin()
        with self._condition:
            while self._waiting and not self._drained:
                self._condition.wait()
            if self._drained:
                return
            self._waiting = True
        try:
            self._history_activity.wait_until_idle()
            self._target_controller.wait_until_idle()
        except BaseException:
            with self._condition:
                self._waiting = False
                self._condition.notify_all()
            raise
        else:
            with self._condition:
                self._drained = True
                self._waiting = False
                self._condition.notify_all()


@dataclass(frozen=True)
class GUIServerConfig:
    service_url: str = "http://127.0.0.1:8766"
    history_path: Path = Path.home() / ".aegis" / "gui-history.sqlite3"
    history_max_records: int = 1000
    history_max_age_days: float = 30.0
    history_max_bytes: int = 268435456
    api_token: str | None = field(default=None, repr=False)
    admin_token: str | None = field(default=None, repr=False)
    upstream_timeout_seconds: float = 120.0
    project_root: Path = field(default_factory=Path.cwd)
    target_switch_timeout_seconds: float = 1200.0
    max_http_connections: int = 32
    request_read_timeout_seconds: float = 15.0
    session_token: str = field(
        default_factory=lambda: secrets.token_urlsafe(32),
        repr=False,
    )
    shutdown_token: str = field(
        default_factory=lambda: secrets.token_urlsafe(32),
        repr=False,
    )

    def __post_init__(self) -> None:
        _require_positive_finite(
            self.upstream_timeout_seconds,
            "upstream_timeout_seconds",
        )
        _require_positive_int(self.history_max_records, "history_max_records")
        _require_positive_finite(self.history_max_age_days, "history_max_age_days")
        _require_positive_int(self.history_max_bytes, "history_max_bytes")
        _require_positive_finite(
            self.target_switch_timeout_seconds,
            "target_switch_timeout_seconds",
        )
        _require_positive_int(self.max_http_connections, "max_http_connections")
        _require_positive_finite(
            self.request_read_timeout_seconds,
            "request_read_timeout_seconds",
        )
        for name, value in (
            ("session_token", self.session_token),
            ("shutdown_token", self.shutdown_token),
        ):
            if not isinstance(value, str) or re.fullmatch(
                rf"[A-Za-z0-9_-]{{{MIN_SECRET_CHARACTERS},}}",
                value,
            ) is None:
                raise ValueError(
                    f"{name} must contain at least {MIN_SECRET_CHARACTERS} "
                    "URL-safe visible ASCII characters."
                )
        for name, value in (
            ("api_token", self.api_token),
            ("admin_token", self.admin_token),
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string when configured.")
            if value is not None and any(
                character in value for character in ("\r", "\n")
            ):
                raise ValueError(f"{name} must not contain line breaks.")
        scoped_secrets = {
            "session_token": self.session_token,
            "shutdown_token": self.shutdown_token,
            "api_token": self.api_token,
            "admin_token": self.admin_token,
        }
        secret_problems = validate_distinct_secrets(scoped_secrets)
        if secret_problems:
            raise ValueError(" ".join(secret_problems))


class GuardrailUpstream:
    def __init__(
        self,
        service_url: str,
        *,
        api_token: str | None,
        admin_token: str | None,
        timeout_seconds: float,
    ) -> None:
        _require_positive_finite(timeout_seconds, "timeout_seconds")
        self._service_url = _validate_service_url(service_url)
        self._api_token = api_token
        self._admin_token = admin_token
        self._timeout_seconds = float(timeout_seconds)
        parsed = urlsplit(self._service_url)
        self._credentialed_loopback = bool(
            parsed.hostname and _is_loopback_host(parsed.hostname)
        )
        self._local_credential_guard: (
            Callable[[str | None], AbstractContextManager[str]] | None
        ) = None

    @property
    def service_url(self) -> str:
        return self._service_url

    def set_local_credential_guard(
        self,
        guard: Callable[[str | None], AbstractContextManager[str]] | None,
    ) -> None:
        """Install the port-bound Compose ownership proof for local bearers."""

        self._local_credential_guard = guard

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace_id = str(uuid.uuid4())
        body = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Request-ID": trace_id,
        }
        expected_target = None
        if self._api_token is not None:
            headers["Authorization"] = f"Bearer {self._api_token}"
            if self._credentialed_loopback:
                expected_target = _credential_target(self.readiness())
        with self._credential_scope(expected_target, self._api_token):
            response, response_headers = self._request_json(
                "/v1/guard",
                method="POST",
                data=body,
                headers=headers,
                timeout=self._timeout_seconds,
            )
        if response.get("trace_id") != trace_id:
            raise UpstreamGuardrailError(
                "The AEGIS service returned a mismatched trace identifier.",
                code="invalid_upstream_response",
            )
        if response_headers.get("X-Request-ID") != trace_id:
            raise UpstreamGuardrailError(
                "The AEGIS service returned an invalid trace header.",
                code="invalid_upstream_response",
            )
        decisions = response.get("decisions")
        summary = response.get("summary")
        if not isinstance(decisions, list) or not decisions or not isinstance(summary, dict):
            raise UpstreamGuardrailError(
                "The AEGIS service returned an invalid decision payload.",
                code="invalid_upstream_response",
            )
        return response

    def status(self) -> dict[str, Any]:
        try:
            readiness = self.readiness()
        except UpstreamGuardrailError as exc:
            return {
                "connected": False,
                "ready": False,
                "error": {"code": exc.code, "message": str(exc)},
                "readiness": None,
                "capabilities": None,
                "metrics": None,
            }

        metrics = None
        metrics_error = None
        authenticated_readiness = readiness
        authenticated_headers = {"Accept": "application/json"}
        expected_target = None
        if self._api_token is not None:
            authenticated_headers["Authorization"] = f"Bearer {self._api_token}"
        try:
            if self._api_token is not None and self._credentialed_loopback:
                expected_target = _credential_target(readiness)
            with self._credential_scope(expected_target, self._api_token):
                if self._api_token is not None:
                    authenticated_readiness, _ = self._request_json(
                        "/v1/status",
                        method="GET",
                        headers=authenticated_headers,
                        timeout=min(self._timeout_seconds, 5.0),
                        accepted_statuses={
                            HTTPStatus.OK,
                            HTTPStatus.SERVICE_UNAVAILABLE,
                        },
                    )
                metrics, _ = self._request_json(
                    "/metrics",
                    method="GET",
                    headers=authenticated_headers,
                    timeout=min(self._timeout_seconds, 5.0),
                )
        except UpstreamGuardrailError as exc:
            metrics_error = {"code": exc.code, "message": str(exc)}
        return {
            "connected": True,
            "ready": bool(authenticated_readiness.get("ok")),
            "error": None,
            "readiness": authenticated_readiness,
            "capabilities": _readiness_capabilities(authenticated_readiness),
            "metrics": metrics,
            "metrics_error": metrics_error,
        }

    def readiness(self) -> dict[str, Any]:
        readiness, _ = self._request_json(
            "/readyz",
            method="GET",
            headers={"Accept": "application/json"},
            timeout=min(self._timeout_seconds, 5.0),
            accepted_statuses={HTTPStatus.OK, HTTPStatus.SERVICE_UNAVAILABLE},
        )
        return readiness

    def authenticated_status(
        self,
        *,
        expected_target: str,
    ) -> dict[str, Any]:
        """Read the full runtime identity only inside the local owner guard."""

        if self._api_token is None:
            raise UpstreamGuardrailError(
                "The authenticated runtime status is unavailable without an API token.",
                code="runtime_status_authentication_unavailable",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_token}",
        }
        with self._credential_scope(expected_target, self._api_token):
            response, _ = self._request_json(
                "/v1/status",
                method="GET",
                headers=headers,
                timeout=min(self._timeout_seconds, 5.0),
                accepted_statuses={
                    HTTPStatus.OK,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                },
            )
        return response

    def input_capabilities(self) -> tuple[tuple[str, ...] | None, str | None]:
        readiness = self.readiness()
        capabilities = _readiness_capabilities(readiness)
        modalities = capabilities.get("input_modalities")
        target_profile = readiness.get("target_profile")
        return (
            None
            if modalities is None
            else tuple(str(item) for item in modalities),
            target_profile if isinstance(target_profile, str) else None,
        )

    def set_traffic_mode(
        self,
        traffic_mode: str,
        *,
        expected_target: str,
    ) -> dict[str, Any]:
        body = json.dumps(
            {"traffic_mode": str(traffic_mode)},
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._admin_token is not None:
            headers["Authorization"] = f"Bearer {self._admin_token}"
        with self._credential_scope(expected_target, self._admin_token):
            response, _ = self._request_json(
                "/v1/admin/traffic-mode",
                method="PUT",
                data=body,
                headers=headers,
                timeout=min(self._timeout_seconds, 10.0),
            )
        if response.get("traffic_mode") != traffic_mode:
            raise UpstreamGuardrailError(
                "The AEGIS service did not apply the requested traffic mode.",
                code="invalid_upstream_response",
            )
        return response

    @contextmanager
    def _credential_scope(
        self,
        expected_target: str | None,
        token: str | None,
    ) -> Iterator[None]:
        if token is None or not self._credentialed_loopback:
            yield
            return
        guard = self._local_credential_guard
        if guard is None:
            raise UpstreamGuardrailError(
                (
                    "The local AEGIS bearer was not sent because listener "
                    "ownership could not be proven."
                ),
                code="credential_owner_unverified",
                status=HTTPStatus.BAD_GATEWAY,
            )
        entered = False
        try:
            with guard(expected_target):
                entered = True
                yield
        except GUIControlError as exc:
            # Request processing does not raise GUIControlError, so an error before
            # entering the scope is an ownership/lifecycle failure from the guard.
            if entered:
                raise
            raise UpstreamGuardrailError(
                str(exc),
                code=exc.code,
                status=HTTPStatus.BAD_GATEWAY,
            ) from None

    def _request_json(
        self,
        path: str,
        *,
        method: str,
        headers: dict[str, str],
        timeout: float,
        data: bytes | None = None,
        accepted_statuses: set[HTTPStatus] | None = None,
    ) -> tuple[dict[str, Any], Any]:
        accepted = {int(item) for item in (accepted_statuses or {HTTPStatus.OK})}
        try:
            response = request_with_deadline(
                self._service_url,
                path,
                method=method,
                headers=headers,
                data=data,
                timeout_seconds=timeout,
                max_response_bytes=_MAX_UPSTREAM_RESPONSE_BYTES,
            )
            status = response.status
            response_headers = response.headers
            content_type = response.headers.get_content_type()
            body = response.body
        except (OSError, TimeoutError, ValueError):
            raise UpstreamGuardrailError(
                "The AEGIS service could not be reached.",
                code="service_unavailable",
                status=HTTPStatus.BAD_GATEWAY,
            ) from None

        if status not in accepted:
            raise _upstream_http_error(status, content_type, body)
        if content_type != "application/json" or len(body) > _MAX_UPSTREAM_RESPONSE_BYTES:
            raise UpstreamGuardrailError(
                "The AEGIS service returned an invalid response.",
                code="invalid_upstream_response",
            )
        try:
            payload = strict_json_loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise UpstreamGuardrailError(
                "The AEGIS service returned unreadable JSON.",
                code="invalid_upstream_response",
            ) from None
        if not isinstance(payload, dict):
            raise UpstreamGuardrailError(
                "The AEGIS service returned an invalid JSON payload.",
                code="invalid_upstream_response",
            )
        return payload, response_headers


def create_gui_http_server(
    host: str,
    port: int,
    config: GUIServerConfig,
    *,
    target_controller: ComposeTargetController | None = None,
) -> BoundedThreadingHTTPServer:
    if not _is_loopback_host(host):
        raise ValueError("The AEGIS GUI may only bind to a loopback host.")
    history = GUIHistoryStore(
        config.history_path,
        max_records=config.history_max_records,
        max_age_days=config.history_max_age_days,
        max_bytes=config.history_max_bytes,
    )
    upstream = GuardrailUpstream(
        config.service_url,
        api_token=config.api_token,
        admin_token=config.admin_token,
        timeout_seconds=config.upstream_timeout_seconds,
    )
    controller = target_controller or ComposeTargetController(
        config.project_root,
        upstream,
        startup_timeout_seconds=config.target_switch_timeout_seconds,
    )
    credential_guard = getattr(controller, "credentialed_request", None)
    upstream.set_local_credential_guard(
        credential_guard if callable(credential_guard) else None
    )
    history_activity = _HistoryActivityBarrier()
    lifecycle = _GUILifecycle(history_activity, controller)
    assets = _load_static_assets()
    handler = _gui_handler_class(
        history,
        history_activity,
        upstream,
        controller,
        lifecycle,
        config.shutdown_token,
        config.session_token,
        config.admin_token is not None,
        assets,
    )
    try:
        address = ipaddress.ip_address(str(host).strip())
    except ValueError:
        address = None
    server_class = (
        BoundedIPv6ThreadingHTTPServer
        if isinstance(address, ipaddress.IPv6Address)
        else BoundedThreadingHTTPServer
    )
    server = server_class(
        (host, port),
        handler,
        max_http_connections=config.max_http_connections,
        request_read_timeout_seconds=config.request_read_timeout_seconds,
    )
    server.daemon_threads = True
    server.begin_drain_and_wait = lifecycle.begin_and_wait  # type: ignore[attr-defined]
    return server


def _gui_handler_class(
    history: GUIHistoryStore,
    history_activity: _HistoryActivityBarrier,
    upstream: GuardrailUpstream,
    target_controller: ComposeTargetController,
    lifecycle: _GUILifecycle,
    shutdown_token: str,
    session_token: str,
    runtime_admin_token_configured: bool,
    assets: dict[str, tuple[bytes, str]],
):
    settings_lock = Lock()
    traffic_mode_preferences: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        server_version = "AEGIS-GUI"
        sys_version = ""

        def do_GET(self) -> None:
            if self._request_content_length(body_required=False) is None:
                return
            if not self._trusted_host():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "host_mismatch",
                    "The dashboard is available only through a loopback address.",
                )
                return
            parsed = urlsplit(self.path)
            path = parsed.path
            if path in assets:
                body, content_type = assets[path]
                self._send_bytes(
                    body,
                    status=HTTPStatus.OK,
                    content_type=content_type,
                )
                return
            if path.startswith("/api/") and not self._authorized_session():
                return
            if path == "/api/status":
                self._status()
                return
            if path == "/api/settings":
                self._settings()
                return
            if path == "/api/history/summary":
                self._history_summary()
                return
            if path == "/api/history":
                self._history_search(parse_qs(parsed.query, keep_blank_values=False))
                return
            image_match = re.fullmatch(
                r"/api/history/([0-9a-f-]{36})/images/([0-9]+)",
                path,
            )
            if image_match:
                self._history_image(image_match.group(1), int(image_match.group(2)))
                return
            detail_match = re.fullmatch(r"/api/history/([0-9a-f-]{36})", path)
            if detail_match:
                self._history_detail(detail_match.group(1))
                return
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                "The requested GUI resource does not exist.",
            )

        def do_POST(self) -> None:
            parsed = urlsplit(self.path)
            body_required = parsed.path == "/api/evaluate" and not parsed.query
            length = self._request_content_length(body_required=body_required)
            if length is None:
                return
            if not self._trusted_host():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "host_mismatch",
                    "The dashboard is available only through a loopback address.",
                )
                return
            if parsed.path == "/api/shutdown" and not parsed.query:
                self._shutdown()
                return
            if parsed.path.startswith("/api/") and not self._authorized_session():
                return
            if parsed.path != "/api/evaluate" or parsed.query:
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "not_found",
                    "Use POST /api/evaluate or POST /api/shutdown.",
                )
                return
            if not self._same_origin():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "origin_mismatch",
                    "The request origin does not match the GUI.",
                )
                return
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != "application/json":
                self._error(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "unsupported_media_type",
                    "Content-Type must be application/json.",
                )
                return
            if length > _MAX_GUI_BODY_BYTES:
                self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    f"GUI request exceeds {_MAX_GUI_BODY_BYTES} bytes.",
                )
                return
            encoded_body = self._read_request_body(length)
            if encoded_body is None:
                return
            try:
                raw = strict_json_loads(encoded_body.decode("utf-8"))
                payload, images = _normalize_submission(raw)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                GUISubmissionError,
                ValueError,
            ) as exc:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_submission",
                    str(exc),
                )
                return

            try:
                with history_activity.evaluation():
                    try:
                        supported_modalities, target_profile = (
                            upstream.input_capabilities()
                        )
                    except UpstreamGuardrailError:
                        # Older or temporarily unavailable services may not advertise a
                        # contract. Preserve the existing forwarding behavior so the
                        # upstream remains authoritative in that case.
                        supported_modalities = None
                        target_profile = None
                    modality = _submission_modality(payload)
                    if (
                        supported_modalities is not None
                        and modality not in supported_modalities
                    ):
                        self._error(
                            HTTPStatus.UNPROCESSABLE_ENTITY,
                            "unsupported_modality",
                            _unsupported_modality_message(
                                modality,
                                supported_modalities,
                            ),
                            error_details={
                                "modality": modality,
                                "supported_modalities": list(supported_modalities),
                                "target_profile": target_profile,
                            },
                        )
                        return
                    self._evaluate_submission(payload, images)
            except GUIShuttingDownError:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "shutting_down",
                    "The dashboard is shutting down and cannot accept evaluations.",
                )
            except GUITargetChangingError:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "target_change_in_progress",
                    "The active detector target is changing; retry when it is ready.",
                )

        def _shutdown(self) -> None:
            authorization = self.headers.get("Authorization", "")
            if not (
                authorization.startswith("Bearer ")
                and secret_matches(authorization[7:], shutdown_token)
            ):
                self._error(
                    HTTPStatus.UNAUTHORIZED,
                    "unauthorized",
                    "A valid bearer token is required to stop the GUI.",
                    extra={"WWW-Authenticate": "Bearer"},
                )
                return
            if self.headers.get("Origin") and not self._same_origin():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "origin_mismatch",
                    "The request origin does not match the GUI.",
                )
                return
            if self.headers.get("X-AEGIS-Confirmation") != "shutdown-gui":
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "confirmation_required",
                    "Stopping the GUI requires explicit confirmation.",
                )
                return
            already_requested = lifecycle.begin()
            self._send_json(
                {
                    "shutting_down": True,
                    "already_requested": already_requested,
                },
                status=HTTPStatus.ACCEPTED,
            )
            if already_requested:
                return

            def finish_shutdown() -> None:
                lifecycle.begin_and_wait()
                self.server.shutdown()

            Thread(
                target=finish_shutdown,
                name="aegis-gui-shutdown",
                daemon=True,
            ).start()

        def do_PUT(self) -> None:
            parsed = urlsplit(self.path)
            body_required = parsed.path == "/api/settings" and not parsed.query
            length = self._request_content_length(body_required=body_required)
            if length is None:
                return
            if not self._trusted_host():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "host_mismatch",
                    "The dashboard is available only through a loopback address.",
                )
                return
            if parsed.path.startswith("/api/") and not self._authorized_session():
                return
            if parsed.path != "/api/settings" or parsed.query:
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "not_found",
                    "Use PUT /api/settings to update GUI service settings.",
                )
                return
            if not self._strict_same_origin():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "origin_mismatch",
                    "A same-origin request is required to change service settings.",
                )
                return
            if lifecycle.requested:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "shutting_down",
                    "The dashboard is shutting down and cannot change targets.",
                )
                return
            payload = self._read_settings_payload(length)
            if payload is None:
                return

            unknown = set(payload) - {"target_profile", "traffic_mode"}
            if unknown:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_settings",
                    f"Unsupported settings fields: {sorted(unknown)}.",
                )
                return
            target_profile = payload.get("target_profile")
            traffic_mode = payload.get("traffic_mode")
            if not isinstance(target_profile, str) or not target_profile:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_target",
                    "target_profile must name an available target.",
                )
                return
            if traffic_mode not in _TRAFFIC_MODES:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_traffic_mode",
                    f"traffic_mode must be one of {list(_TRAFFIC_MODES)}.",
                )
                return

            service_status = upstream.status()
            if not _authenticated_runtime_control(
                service_status,
                admin_token_configured=runtime_admin_token_configured,
            ):
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "runtime_control_unavailable",
                    (
                        "Runtime settings require a running AEGIS service and a "
                        "matching API bearer token; applying changes also requires "
                        "the distinct matching admin bearer token in the GUI process."
                    ),
                )
                return
            readiness = service_status.get("readiness")
            previous_target = (
                readiness.get("target_profile")
                if isinstance(readiness, dict)
                and isinstance(readiness.get("target_profile"), str)
                else None
            )
            try:
                if previous_target == target_profile:
                    operation = target_controller.switch(
                        target_profile,
                        previous_target,
                        traffic_mode,
                    )
                else:
                    with history_activity.target_change():
                        operation = target_controller.switch(
                            target_profile,
                            previous_target,
                            traffic_mode,
                        )
            except GUIControlBusyError as exc:
                self._error(
                    HTTPStatus.CONFLICT,
                    exc.code,
                    str(exc),
                )
                return
            except GUIControlError as exc:
                status = (
                    HTTPStatus.BAD_REQUEST
                    if exc.code == "invalid_target"
                    else HTTPStatus.SERVICE_UNAVAILABLE
                )
                self._error(
                    status,
                    exc.code,
                    str(exc),
                    error_details={
                        "rollback_succeeded": exc.rollback_succeeded,
                    },
                )
                return
            except UpstreamGuardrailError as exc:
                status = (
                    HTTPStatus.BAD_GATEWAY
                    if exc.status < 400 or exc.status > 599
                    else HTTPStatus(exc.status)
                )
                self._error(status, exc.code, str(exc))
                return
            except GUIShuttingDownError:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "shutting_down",
                    "The dashboard is shutting down and cannot change targets.",
                )
                return

            with settings_lock:
                traffic_mode_preferences[target_profile] = traffic_mode
            response = self._settings_payload()
            response["operation"] = operation
            self._send_json(response, status=HTTPStatus.OK)

        def do_DELETE(self) -> None:
            if self._request_content_length(body_required=False) is None:
                return
            if not self._trusted_host():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "host_mismatch",
                    "The dashboard is available only through a loopback address.",
                )
                return
            parsed = urlsplit(self.path)
            if parsed.path.startswith("/api/") and not self._authorized_session():
                return
            if parsed.path != "/api/history" or parsed.query:
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "not_found",
                    "Use DELETE /api/history to clear all GUI history.",
                )
                return
            if not self._strict_same_origin():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "origin_mismatch",
                    "A same-origin request is required to clear GUI history.",
                )
                return
            if self.headers.get("X-AEGIS-Confirmation") != "clear-history":
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "confirmation_required",
                    "Clearing history requires explicit confirmation.",
                )
                return
            try:
                cleared = history_activity.clear(history.clear_all)
            except GUIShuttingDownError:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "shutting_down",
                    "The dashboard is shutting down and cannot clear history.",
                )
                return
            except (OSError, sqlite3.Error, ValueError):
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "history_unavailable",
                    "The GUI history database is unavailable.",
                )
                return
            self._send_json(
                {"cleared": True, **cleared},
                status=HTTPStatus.OK,
            )

        def log_message(self, format: str, *args) -> None:
            return

        def _evaluate_submission(
            self,
            payload: dict[str, Any],
            images: tuple[StoredImage, ...],
        ) -> None:
            try:
                history_id = history.create_pending(
                    text=str(payload.get("text", "")),
                    request_id=str(payload["request_id"]),
                    metadata=dict(payload.get("metadata", {})),
                    images=images,
                )
            except GUIHistoryRetentionError as exc:
                self._error(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    "history_capacity",
                    str(exc),
                )
                return
            except (OSError, sqlite3.Error, ValueError):
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "history_unavailable",
                    "The GUI history database is unavailable.",
                )
                return

            started = time.perf_counter()
            try:
                result = upstream.evaluate(payload)
                duration_ms = (time.perf_counter() - started) * 1000.0
                history.complete(history_id, result, duration_ms)
            except UpstreamGuardrailError as exc:
                duration_ms = (time.perf_counter() - started) * 1000.0
                error = {
                    "code": exc.code,
                    "message": str(exc),
                    "status": exc.status,
                }
                try:
                    history.fail(history_id, error, duration_ms)
                except (OSError, sqlite3.Error, ValueError):
                    pass
                status = (
                    HTTPStatus.BAD_GATEWAY
                    if exc.status < 400 or exc.status > 599
                    else HTTPStatus(exc.status)
                )
                self._send_json(
                    {"error": error, "history_id": history_id},
                    status=status,
                )
                return
            except (OSError, sqlite3.Error, ValueError):
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "history_unavailable",
                    "The decision was produced, but GUI history could not be updated.",
                    extra={"history_id": history_id},
                )
                return

            self._send_json(
                {
                    "history_id": history_id,
                    "result": result,
                    "duration_ms": duration_ms,
                },
                status=HTTPStatus.OK,
            )

        def _status(self) -> None:
            try:
                history_summary = history.summary()
            except (OSError, sqlite3.Error, ValueError):
                history_summary = None
            self._send_json(
                {
                    "service": {
                        "url": upstream.service_url,
                        **upstream.status(),
                    },
                    "history": history_summary,
                    "limits": {
                        "max_text_characters": _MAX_TEXT_CHARACTERS,
                        "max_image_bytes": _MAX_IMAGE_BYTES,
                        "max_image_pixels": _MAX_IMAGE_PIXELS,
                        "allowed_image_media_types": sorted(_ALLOWED_IMAGE_TYPES),
                    },
                },
                status=HTTPStatus.OK,
            )

        def _settings(self) -> None:
            self._send_json(
                self._settings_payload(),
                status=HTTPStatus.OK,
            )

        def _settings_payload(self) -> dict[str, object]:
            service_status = upstream.status()
            readiness = service_status.get("readiness")
            readiness = readiness if isinstance(readiness, dict) else {}
            active_target = (
                readiness.get("target_profile")
                if isinstance(readiness.get("target_profile"), str)
                else None
            )
            traffic_mode = (
                readiness.get("traffic_mode")
                if readiness.get("traffic_mode") in _TRAFFIC_MODES
                else None
            )
            if active_target is not None and traffic_mode is not None:
                with settings_lock:
                    traffic_mode_preferences[active_target] = traffic_mode

            control = target_controller.metadata(active_target)
            raw_targets = control.get("targets")
            targets: list[dict[str, object]] = []
            for raw_target in raw_targets if isinstance(raw_targets, list) else []:
                if not isinstance(raw_target, dict):
                    continue
                target = dict(raw_target)
                name = target.get("name", target.get("id"))
                if not isinstance(name, str):
                    continue
                target["name"] = name
                target.pop("id", None)
                with settings_lock:
                    target_mode = traffic_mode_preferences.get(name, "shadow")
                if name == active_target and traffic_mode is not None:
                    target_mode = traffic_mode
                target["traffic_mode"] = target_mode
                targets.append(target)

            target_control = {
                name: control.get(name)
                for name in ("available", "busy", "reason", "operation")
            }
            return {
                "current": {
                    "target_profile": active_target,
                    "traffic_mode": traffic_mode,
                    "model_id": readiness.get("model_id"),
                    "detector_mode": readiness.get("detector_mode", "single"),
                    "detector_identity_sha256": readiness.get(
                        "detector_identity_sha256", readiness.get("detector_sha256")
                    ),
                    "ready": bool(service_status.get("ready")),
                    "connected": bool(service_status.get("connected")),
                    "runtime_traffic_mode_control": (
                        _authenticated_runtime_control(
                            service_status,
                            admin_token_configured=runtime_admin_token_configured,
                        )
                    ),
                },
                "targets": targets,
                "traffic_modes": list(_TRAFFIC_MODES),
                "target_control": target_control,
            }

        def _read_settings_payload(self, length: int) -> dict[str, Any] | None:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != "application/json":
                self._error(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "unsupported_media_type",
                    "Content-Type must be application/json.",
                )
                return None
            if length > 32 * 1024:
                self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    "The settings request is too large.",
                )
                return None
            encoded_body = self._read_request_body(length)
            if encoded_body is None:
                return None
            try:
                payload = strict_json_loads(encoded_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_settings",
                    "The settings request must contain valid JSON.",
                )
                return None
            if not isinstance(payload, dict):
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_settings",
                    "The settings request must be a JSON object.",
                )
                return None
            return payload

        def _history_summary(self) -> None:
            try:
                payload = history.summary()
            except (OSError, sqlite3.Error, ValueError):
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "history_unavailable",
                    "The GUI history database is unavailable.",
                )
                return
            self._send_json(payload, status=HTTPStatus.OK)

        def _history_search(self, query: dict[str, list[str]]) -> None:
            try:
                limit = _bounded_integer(_first(query, "limit"), default=25, maximum=100)
                offset = _bounded_integer(
                    _first(query, "offset"),
                    default=0,
                    maximum=1_000_000,
                )
                search = (_first(query, "q") or "").strip()
                if len(search) > 256:
                    raise ValueError("Search text must be at most 256 characters.")
                date_from = _validated_date(_first(query, "from"), "from")
                date_to = _validated_date(_first(query, "to"), "to")
                payload = history.search(
                    query=search,
                    action=_first(query, "action"),
                    verdict=_first(query, "verdict"),
                    modality=_first(query, "modality"),
                    status=_first(query, "status"),
                    date_from=date_from,
                    date_to=date_to,
                    limit=limit,
                    offset=offset,
                )
            except ValueError as exc:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_filter",
                    str(exc),
                )
                return
            except (OSError, sqlite3.Error):
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "history_unavailable",
                    "The GUI history database is unavailable.",
                )
                return
            self._send_json(payload, status=HTTPStatus.OK)

        def _history_detail(self, history_id: str) -> None:
            try:
                payload = history.get(history_id)
            except (OSError, sqlite3.Error, ValueError):
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "history_unavailable",
                    "The GUI history database is unavailable.",
                )
                return
            if payload is None:
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "history_not_found",
                    "That history record does not exist.",
                )
                return
            images = payload.get("images")
            if isinstance(images, list):
                for item in images:
                    if isinstance(item, dict) and isinstance(item.get("position"), int):
                        item["url"] = (
                            f"/api/history/{history_id}/images/{item['position']}"
                        )
            input_payload = payload.get("input")
            if isinstance(input_payload, dict) and "images" not in input_payload:
                input_payload["images"] = images if isinstance(images, list) else []
            self._send_json(payload, status=HTTPStatus.OK)

        def _history_image(self, history_id: str, position: int) -> None:
            try:
                image = history.get_image(history_id, position)
            except (OSError, sqlite3.Error, ValueError):
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "history_unavailable",
                    "The GUI history database is unavailable.",
                )
                return
            if image is None:
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "image_not_found",
                    "That history image does not exist.",
                )
                return
            self._send_bytes(
                image.content,
                status=HTTPStatus.OK,
                content_type=image.media_type,
                extra_headers={
                    "Content-Disposition": (
                        f'inline; filename="{_safe_header_filename(image.filename)}"'
                    )
                },
            )

        def _same_origin(self) -> bool:
            origin = self.headers.get("Origin")
            if not origin:
                return True
            parsed = urlsplit(origin)
            return (
                parsed.scheme == "http"
                and parsed.hostname is not None
                and _is_loopback_host(parsed.hostname)
                and parsed.netloc == self.headers.get("Host", "")
            )

        def _strict_same_origin(self) -> bool:
            return bool(self.headers.get("Origin")) and self._same_origin()

        def _trusted_host(self) -> bool:
            host = self.headers.get("Host", "")
            try:
                hostname = urlsplit(f"http://{host}").hostname
            except ValueError:
                return False
            return hostname is not None and _is_loopback_host(hostname)

        def _authorized_session(self) -> bool:
            candidates = self.headers.get_all(_GUI_SESSION_HEADER, [])
            if (
                len(candidates) == 1
                and secret_matches(candidates[0], session_token)
            ):
                return True
            self._error(
                HTTPStatus.UNAUTHORIZED,
                "gui_session_required",
                "A valid per-launch GUI session header is required.",
            )
            return False

        def _error(
            self,
            status: HTTPStatus,
            code: str,
            message: str,
            *,
            extra: dict[str, object] | None = None,
            error_details: dict[str, object] | None = None,
        ) -> None:
            error: dict[str, object] = {"code": code, "message": message}
            if error_details:
                error.update(error_details)
            payload: dict[str, object] = {"error": error}
            if extra:
                payload.update(extra)
            self._send_json(payload, status=status)

        def _send_json(self, payload: object, *, status: HTTPStatus) -> None:
            body = json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._send_bytes(
                body,
                status=status,
                content_type="application/json",
            )

        def _send_bytes(
            self,
            body: bytes,
            *,
            status: HTTPStatus,
            content_type: str,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self._finish_request_read()
            try:
                self.send_response(int(status))
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for name, value in _SECURITY_HEADERS.items():
                    self.send_header(name, value)
                for name, value in (extra_headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)
            except (OSError, socket.timeout):
                self.close_connection = True

        def _request_content_length(self, *, body_required: bool) -> int | None:
            try:
                return parse_request_content_length(
                    self.headers,
                    body_required=body_required,
                )
            except RequestFramingError as exc:
                self.close_connection = True
                length_required = body_required and exc.reason in {
                    "missing_content_length",
                    "invalid_content_length",
                    "body_required",
                }
                self._error(
                    (
                        HTTPStatus.LENGTH_REQUIRED
                        if length_required
                        else HTTPStatus.BAD_REQUEST
                    ),
                    (
                        "content_length_required"
                        if length_required
                        else "invalid_request"
                    ),
                    str(exc),
                )
                return None

        def _read_request_body(self, length: int) -> bytes | None:
            try:
                body = self.rfile.read(length)
            except (OSError, TimeoutError, socket.timeout):
                self._finish_request_read()
                self._error(
                    HTTPStatus.REQUEST_TIMEOUT,
                    "request_timeout",
                    "The request body was not received within the configured timeout.",
                )
                return None
            expired = self._finish_request_read()
            if expired or len(body) != length:
                self._error(
                    HTTPStatus.REQUEST_TIMEOUT,
                    "request_timeout",
                    "The request body was not received within the configured timeout.",
                )
                return None
            return body

        def _finish_request_read(self) -> bool:
            finish = getattr(self.server, "finish_request_read", None)
            return bool(finish(self.connection)) if callable(finish) else False

    return Handler


def _normalize_submission(
    raw: object,
) -> tuple[dict[str, Any], tuple[StoredImage, ...]]:
    if not isinstance(raw, dict):
        raise GUISubmissionError("The GUI submission must be a JSON object.")
    unknown = set(raw) - {"text", "image"}
    if unknown:
        raise GUISubmissionError(f"Unsupported GUI fields: {sorted(unknown)}.")
    text = raw.get("text", "")
    if not isinstance(text, str):
        raise GUISubmissionError("Prompt text must be a string.")
    if len(text) > _MAX_TEXT_CHARACTERS:
        raise GUISubmissionError(
            f"Prompt text exceeds {_MAX_TEXT_CHARACTERS} characters."
        )

    images: tuple[StoredImage, ...] = ()
    upstream_images: list[dict[str, str]] = []
    image = raw.get("image")
    if image is not None:
        stored = _decode_gui_image(image)
        images = (stored,)
        upstream_images.append(
            {
                "media_type": stored.media_type,
                "base64": base64.b64encode(stored.content).decode("ascii"),
            }
        )
    if not text.strip() and not images:
        raise GUISubmissionError("Enter prompt text, attach an image, or provide both.")

    request_id = f"gui-{uuid.uuid4()}"
    payload: dict[str, Any] = {
        "text": text,
        "request_id": request_id,
        "metadata": {"source": "aegis_gui"},
    }
    if upstream_images:
        payload["images"] = upstream_images
    return payload, images


def _submission_modality(payload: dict[str, Any]) -> str:
    has_text = bool(str(payload.get("text", "")).strip())
    images = payload.get("images")
    has_image = isinstance(images, list) and bool(images)
    if has_text and has_image:
        return "image_text"
    if has_image:
        return "image"
    return "text"


def _authenticated_runtime_control(
    service_status: object,
    *,
    admin_token_configured: bool,
) -> bool:
    if not isinstance(service_status, dict):
        return False
    capabilities = service_status.get("capabilities")
    return bool(
        admin_token_configured
        and isinstance(capabilities, dict)
        and capabilities.get("runtime_traffic_mode_control") is True
        and isinstance(service_status.get("metrics"), dict)
        and service_status.get("metrics_error") is None
    )


def _readiness_capabilities(readiness: object) -> dict[str, object]:
    if not isinstance(readiness, dict):
        return {"input_modalities": None}
    raw_capabilities = readiness.get("capabilities")
    capabilities = raw_capabilities if isinstance(raw_capabilities, dict) else {}
    raw_modalities = (
        capabilities.get("input_modalities")
        if capabilities
        else readiness.get("supported_modalities")
    )
    normalized_modalities: list[str] | None = None
    if isinstance(raw_modalities, (list, tuple)) and raw_modalities:
        candidates = [str(item) for item in raw_modalities]
        if all(
            value in _INPUT_MODALITIES
            and candidates.index(value) == index
            for index, value in enumerate(candidates)
        ):
            normalized_modalities = candidates

    raw_modes = capabilities.get("traffic_modes")
    normalized_modes: list[str] | None = None
    if isinstance(raw_modes, (list, tuple)) and raw_modes:
        candidates = [str(item) for item in raw_modes]
        if all(
            value in _TRAFFIC_MODES
            and candidates.index(value) == index
            for index, value in enumerate(candidates)
        ):
            normalized_modes = candidates

    return {
        "input_modalities": normalized_modalities,
        "traffic_modes": normalized_modes,
        "runtime_traffic_mode_control": (
            capabilities.get("runtime_traffic_mode_control") is True
        ),
    }


def _unsupported_modality_message(
    modality: str,
    supported_modalities: tuple[str, ...],
) -> str:
    if modality == "text" and "image_text" in supported_modalities:
        instruction = "Attach one image and keep the prompt text."
    elif modality == "image" and "image_text" in supported_modalities:
        instruction = "Add prompt text to evaluate this image."
    else:
        instruction = (
            "Use "
            + _format_modalities(supported_modalities)
            + " with the active target."
        )
    return (
        f"{_modality_label(modality).capitalize()} input is not supported by "
        f"the active AEGIS target. {instruction}"
    )


def _format_modalities(modalities: tuple[str, ...]) -> str:
    labels = [_modality_label(item) for item in modalities]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return f"{', '.join(labels[:-1])}, or {labels[-1]}"


def _modality_label(modality: str) -> str:
    return {
        "text": "text-only",
        "image": "image-only",
        "image_text": "text with one image",
    }.get(modality, modality)


def _decode_gui_image(raw: object) -> StoredImage:
    if not isinstance(raw, dict):
        raise GUISubmissionError("The attached image must be an object.")
    unknown = set(raw) - {"name", "media_type", "base64"}
    if unknown:
        raise GUISubmissionError(f"Unsupported image fields: {sorted(unknown)}.")
    media_type = raw.get("media_type")
    encoded = raw.get("base64")
    if media_type not in _ALLOWED_IMAGE_TYPES:
        raise GUISubmissionError("Attach a PNG, JPEG, or WebP image.")
    if not isinstance(encoded, str) or not encoded:
        raise GUISubmissionError("The attached image is missing its base64 content.")
    maximum_encoded = ((_MAX_IMAGE_BYTES + 2) // 3) * 4
    if len(encoded) > maximum_encoded + 4:
        raise GUISubmissionError(
            f"The attached image exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)} MiB."
        )
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise GUISubmissionError("The attached image is not valid base64.") from None
    if len(content) > _MAX_IMAGE_BYTES:
        raise GUISubmissionError(
            f"The attached image exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)} MiB."
        )
    try:
        with Image.open(BytesIO(content)) as opened:
            opened.verify()
        with Image.open(BytesIO(content)) as opened:
            width, height = opened.size
            actual_format = str(opened.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise GUISubmissionError("The attached file is not a valid supported image.") from None
    if actual_format != _ALLOWED_IMAGE_TYPES[str(media_type)]:
        raise GUISubmissionError("The image media type does not match its content.")
    if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
        raise GUISubmissionError(
            f"The image dimensions exceed {_MAX_IMAGE_PIXELS} pixels."
        )
    filename = _safe_filename(raw.get("name"), media_type=str(media_type))
    return StoredImage(
        filename=filename,
        media_type=str(media_type),
        content=content,
    )


def _load_static_assets() -> dict[str, tuple[bytes, str]]:
    root = files("AEGIS.web")
    return {
        route: (root.joinpath(filename).read_bytes(), content_type)
        for route, (filename, content_type) in _STATIC_ASSETS.items()
    }


def _validate_service_url(value: str) -> str:
    normalized = str(value).strip()
    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise ValueError("service_url is not a valid HTTP(S) origin.") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "service_url must be an HTTP(S) origin without credentials or a path."
        )
    if parsed.scheme == "http" and not _is_loopback_host(hostname):
        raise ValueError("Remote AEGIS services must use HTTPS.")
    return normalized.rstrip("/")


def _is_loopback_host(host: str) -> bool:
    return is_loopback_hostname(host)


def _credential_target(readiness: object) -> str:
    if not isinstance(readiness, dict) or readiness.get("ok") is not True:
        raise UpstreamGuardrailError(
            "The local AEGIS bearer was not sent because the expected target is not ready.",
            code="credential_owner_unverified",
            status=HTTPStatus.BAD_GATEWAY,
        )
    target = readiness.get("target_profile")
    if target not in {"llava05b", "qwen25vl3b"}:
        raise UpstreamGuardrailError(
            "The local AEGIS bearer was not sent because the expected target is invalid.",
            code="credential_owner_unverified",
            status=HTTPStatus.BAD_GATEWAY,
        )
    return str(target)


def _require_positive_int(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_positive_finite(value: object, name: str) -> None:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be positive and finite.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be positive and finite.")


def _upstream_http_error(
    status: int,
    content_type: str,
    body: bytes,
) -> UpstreamGuardrailError:
    code = "upstream_error"
    message = f"AEGIS returned HTTP {status}."
    if content_type == "application/json" and len(body) <= _MAX_UPSTREAM_RESPONSE_BYTES:
        try:
            payload = strict_json_loads(body.decode("utf-8"))
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                raw_code = error.get("code")
                raw_message = error.get("message")
                if isinstance(raw_code, str) and raw_code:
                    code = raw_code
                if isinstance(raw_message, str) and raw_message:
                    message = raw_message
        except (UnicodeDecodeError, ValueError):
            pass
    return UpstreamGuardrailError(message, code=code, status=status)


def _first(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    return None if not values else values[0]


def _bounded_integer(
    value: str | None,
    *,
    default: int,
    maximum: int,
) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError("Pagination values must be integers.") from None
    if parsed < 0 or parsed > maximum:
        raise ValueError(f"Pagination value must be between 0 and {maximum}.")
    return parsed


def _validated_date(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"The {name} date must use YYYY-MM-DD.")
    return value


def _safe_filename(value: object, *, media_type: str) -> str:
    name = "" if value is None else str(value)
    name = re.split(r"[/\\]", name)[-1]
    name = "".join(character for character in name if 32 <= ord(character) < 127)
    name = name.strip(" .")[:160]
    if name:
        return name
    suffix = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }[media_type]
    return f"upload{suffix}"


def _safe_header_filename(value: str) -> str:
    normalized = "".join(
        character
        for character in str(value)
        if 32 <= ord(character) < 127 and character not in {'"', "\\", ";"}
    )
    return normalized[:160] or "image"


__all__ = [
    "GUIServerConfig",
    "GUISubmissionError",
    "GuardrailUpstream",
    "UpstreamGuardrailError",
    "create_gui_http_server",
]
