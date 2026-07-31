from __future__ import annotations

import base64
import binascii
from contextlib import contextmanager
from dataclasses import dataclass, field
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from io import BytesIO
import ipaddress
import json
from pathlib import Path
import re
import socket
import sqlite3
import time
from threading import Condition, Lock, Thread
from typing import Any, Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request
import uuid

from PIL import Image, UnidentifiedImageError

from AEGIS.gui_control import (
    ComposeTargetController,
    GUIControlBusyError,
    GUIControlError,
)
from AEGIS.gui_history import GUIHistoryStore, StoredImage


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


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6


class GUISubmissionError(ValueError):
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

    @contextmanager
    def evaluation(self) -> Iterator[None]:
        with self._condition:
            while self._clearing:
                self._condition.wait()
            self._active_evaluations += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_evaluations -= 1
                self._condition.notify_all()

    def clear(self, operation: Callable[[], dict[str, int]]) -> dict[str, int]:
        with self._condition:
            while self._clearing:
                self._condition.wait()
            self._clearing = True
            while self._active_evaluations:
                self._condition.wait()
        try:
            return operation()
        finally:
            with self._condition:
                self._clearing = False
                self._condition.notify_all()

    def wait_until_idle(self) -> None:
        with self._condition:
            while self._active_evaluations or self._clearing:
                self._condition.wait()


@dataclass(frozen=True)
class GUIServerConfig:
    service_url: str = "http://127.0.0.1:8766"
    history_path: Path = Path.home() / ".aegis" / "gui-history.sqlite3"
    api_token: str | None = None
    upstream_timeout_seconds: float = 120.0
    project_root: Path = field(default_factory=Path.cwd)
    target_switch_timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        if self.upstream_timeout_seconds <= 0:
            raise ValueError("upstream_timeout_seconds must be positive.")
        if self.target_switch_timeout_seconds <= 0:
            raise ValueError("target_switch_timeout_seconds must be positive.")
        if self.api_token is not None and any(
            character in self.api_token for character in ("\r", "\n")
        ):
            raise ValueError("api_token must not contain line breaks.")


class GuardrailUpstream:
    def __init__(
        self,
        service_url: str,
        *,
        api_token: str | None,
        timeout_seconds: float,
    ) -> None:
        self._service_url = _validate_service_url(service_url)
        self._api_token = api_token
        self._timeout_seconds = float(timeout_seconds)
        self._opener = build_opener(_NoRedirectHandler())

    @property
    def service_url(self) -> str:
        return self._service_url

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
        if self._api_token is not None:
            headers["Authorization"] = f"Bearer {self._api_token}"
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
        metric_headers = {"Accept": "application/json"}
        if self._api_token is not None:
            metric_headers["Authorization"] = f"Bearer {self._api_token}"
        try:
            metrics, _ = self._request_json(
                "/metrics",
                method="GET",
                headers=metric_headers,
                timeout=min(self._timeout_seconds, 5.0),
            )
        except UpstreamGuardrailError as exc:
            metrics_error = {"code": exc.code, "message": str(exc)}
        return {
            "connected": True,
            "ready": bool(readiness.get("ok")),
            "error": None,
            "readiness": readiness,
            "capabilities": _readiness_capabilities(readiness),
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

    def set_traffic_mode(self, traffic_mode: str) -> dict[str, Any]:
        body = json.dumps(
            {"traffic_mode": str(traffic_mode)},
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._api_token is not None:
            headers["Authorization"] = f"Bearer {self._api_token}"
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
        request = Request(
            f"{self._service_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        accepted = {int(item) for item in (accepted_statuses or {HTTPStatus.OK})}
        try:
            with self._opener.open(request, timeout=timeout) as response:
                status = int(response.status)
                response_headers = response.headers
                content_type = response.headers.get_content_type()
                body = response.read(_MAX_UPSTREAM_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            status = int(exc.code)
            response_headers = exc.headers
            content_type = exc.headers.get_content_type()
            try:
                body = exc.read(_MAX_UPSTREAM_RESPONSE_BYTES + 1)
            finally:
                exc.close()
            if status not in accepted:
                raise _upstream_http_error(status, content_type, body) from None
        except (OSError, TimeoutError, URLError, ValueError):
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
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
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
) -> ThreadingHTTPServer:
    if not _is_loopback_host(host):
        raise ValueError("The AEGIS GUI may only bind to a loopback host.")
    history = GUIHistoryStore(config.history_path)
    upstream = GuardrailUpstream(
        config.service_url,
        api_token=config.api_token,
        timeout_seconds=config.upstream_timeout_seconds,
    )
    controller = target_controller or ComposeTargetController(
        config.project_root,
        upstream,
        startup_timeout_seconds=config.target_switch_timeout_seconds,
    )
    history_activity = _HistoryActivityBarrier()
    assets = _load_static_assets()
    handler = _gui_handler_class(
        history,
        history_activity,
        upstream,
        controller,
        config.api_token,
        assets,
    )
    try:
        address = ipaddress.ip_address(str(host).strip())
    except ValueError:
        address = None
    server_class = (
        _IPv6ThreadingHTTPServer
        if isinstance(address, ipaddress.IPv6Address)
        else ThreadingHTTPServer
    )
    server = server_class((host, port), handler)
    server.daemon_threads = True
    return server


def _gui_handler_class(
    history: GUIHistoryStore,
    history_activity: _HistoryActivityBarrier,
    upstream: GuardrailUpstream,
    target_controller: ComposeTargetController,
    shutdown_token: str | None,
    assets: dict[str, tuple[bytes, str]],
):
    settings_lock = Lock()
    shutdown_lock = Lock()
    shutdown_state = {"requested": False}
    traffic_mode_preferences: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        server_version = "AEGIS-GUI"
        sys_version = ""

        def do_GET(self) -> None:
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
            if not self._trusted_host():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "host_mismatch",
                    "The dashboard is available only through a loopback address.",
                )
                return
            parsed = urlsplit(self.path)
            if parsed.path == "/api/shutdown" and not parsed.query:
                self._shutdown()
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
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if length <= 0:
                self._error(
                    HTTPStatus.LENGTH_REQUIRED,
                    "content_length_required",
                    "A positive Content-Length is required.",
                )
                return
            if length > _MAX_GUI_BODY_BYTES:
                self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    f"GUI request exceeds {_MAX_GUI_BODY_BYTES} bytes.",
                )
                return
            try:
                raw = json.loads(self.rfile.read(length).decode("utf-8"))
                payload, images = _normalize_submission(raw)
            except (UnicodeDecodeError, json.JSONDecodeError, GUISubmissionError) as exc:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_submission",
                    str(exc),
                )
                return

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

        def _shutdown(self) -> None:
            if shutdown_token is None:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "shutdown_unavailable",
                    "GUI shutdown requires a configured API bearer token.",
                )
                return
            authorization = self.headers.get("Authorization", "")
            expected = f"Bearer {shutdown_token}"
            if not hmac.compare_digest(authorization, expected):
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
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length != 0:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_request",
                    "POST /api/shutdown does not accept a request body.",
                )
                return

            with shutdown_lock:
                already_requested = shutdown_state["requested"]
                shutdown_state["requested"] = True
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
                history_activity.wait_until_idle()
                self.server.shutdown()

            Thread(
                target=finish_shutdown,
                name="aegis-gui-shutdown",
                daemon=True,
            ).start()

        def do_PUT(self) -> None:
            if not self._trusted_host():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "host_mismatch",
                    "The dashboard is available only through a loopback address.",
                )
                return
            parsed = urlsplit(self.path)
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
            payload = self._read_settings_payload()
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
            if not _authenticated_runtime_control(service_status):
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "runtime_control_unavailable",
                    (
                        "Runtime settings require a running AEGIS service and a "
                        "matching API bearer token in the GUI process."
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

            with settings_lock:
                traffic_mode_preferences[target_profile] = traffic_mode
            response = self._settings_payload()
            response["operation"] = operation
            self._send_json(response, status=HTTPStatus.OK)

        def do_DELETE(self) -> None:
            if not self._trusted_host():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "host_mismatch",
                    "The dashboard is available only through a loopback address.",
                )
                return
            parsed = urlsplit(self.path)
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
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length != 0:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_request",
                    "DELETE /api/history does not accept a request body.",
                )
                return
            try:
                cleared = history_activity.clear(history.clear_all)
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
                    "ready": bool(service_status.get("ready")),
                    "connected": bool(service_status.get("connected")),
                    "runtime_traffic_mode_control": (
                        _authenticated_runtime_control(service_status)
                    ),
                },
                "targets": targets,
                "traffic_modes": list(_TRAFFIC_MODES),
                "target_control": target_control,
            }

        def _read_settings_payload(self) -> dict[str, Any] | None:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != "application/json":
                self._error(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "unsupported_media_type",
                    "Content-Type must be application/json.",
                )
                return None
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if length <= 0:
                self._error(
                    HTTPStatus.LENGTH_REQUIRED,
                    "content_length_required",
                    "A positive Content-Length is required.",
                )
                return None
            if length > 32 * 1024:
                self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    "The settings request is too large.",
                )
                return None
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
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


def _authenticated_runtime_control(service_status: object) -> bool:
    if not isinstance(service_status, dict):
        return False
    capabilities = service_status.get("capabilities")
    return bool(
        isinstance(capabilities, dict)
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
    normalized = str(host).strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _upstream_http_error(
    status: int,
    content_type: str,
    body: bytes,
) -> UpstreamGuardrailError:
    code = "upstream_error"
    message = f"AEGIS returned HTTP {status}."
    if content_type == "application/json" and len(body) <= _MAX_UPSTREAM_RESPONSE_BYTES:
        try:
            payload = json.loads(body.decode("utf-8"))
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                raw_code = error.get("code")
                raw_message = error.get("message")
                if isinstance(raw_code, str) and raw_code:
                    code = raw_code
                if isinstance(raw_message, str) and raw_message:
                    message = raw_message
        except (UnicodeDecodeError, json.JSONDecodeError):
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
