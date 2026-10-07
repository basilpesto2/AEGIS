from __future__ import annotations

import csv
import base64
import binascii
import hmac
from http.client import HTTPConnection, RemoteDisconnected
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import struct
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from AEGIS.adapters.common import read_prompt_metadata
from AEGIS.client import (
    GuardrailClient,
    GuardrailRequestError,
    GuardrailTransportError,
    _encode_request,
)
from AEGIS.gui_server import (
    GUIServerConfig,
    GUITargetChangingError,
    GuardrailUpstream,
    UpstreamGuardrailError,
    _HistoryActivityBarrier,
    _authenticated_runtime_control,
    create_gui_http_server,
)
from AEGIS.gui_control import ComposeTargetController
from AEGIS.http_server import ServiceMetrics, create_guardrail_http_server
from AEGIS.http_transport import environment_proxy_for_origin, request_with_deadline
from AEGIS.service import validate_request_payload_schema
from AEGIS.service import RequestLimits, _decode_image_record


class RequestSchemaTests(unittest.TestCase):
    def test_batch_envelope_rejects_unevaluated_sibling_fields(self) -> None:
        payload = {
            "requests": [{"text": "Explain rainbows."}],
            "text": "unevaluated downstream prompt",
            "messages": [{"role": "user", "content": "unevaluated"}],
        }
        with self.assertRaisesRegex(
            GuardrailRequestError,
            "Batch request envelope contains unevaluated fields",
        ):
            _encode_request(payload)
        with self.assertRaisesRegex(
            ValueError,
            "Batch request envelope contains unevaluated fields",
        ):
            validate_request_payload_schema(payload)

    def test_unknown_nested_request_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(GuardrailRequestError, "unknown fields"):
            _encode_request(
                {"requests": [{"text": "benign", "messages": ["malicious"]}]}
            )

    def test_conflicting_image_aliases_are_rejected(self) -> None:
        with self.assertRaisesRegex(GuardrailRequestError, "at most one image source"):
            _encode_request(
                {
                    "text": "prompt",
                    "images": [{"media_type": "image/png", "base64": "AA=="}],
                    "image_base64": "AA==",
                }
            )

    def test_every_cross_source_image_conflict_is_rejected(self) -> None:
        sources = {
            "images": [{"media_type": "image/png", "base64": "AA=="}],
            "image_base64": "AA==",
            "image_path": "untrusted.png",
            "image_paths": ["untrusted.png"],
        }
        names = list(sources)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                with self.subTest(left=left_name, right=right_name):
                    payload = {
                        "text": "prompt",
                        left_name: sources[left_name],
                        right_name: sources[right_name],
                    }
                    with self.assertRaisesRegex(
                        GuardrailRequestError,
                        "at most one image source",
                    ):
                        _encode_request(payload)

    def test_decompression_bomb_is_rejected_as_invalid_input(self) -> None:
        def png_chunk(kind: bytes, data: bytes) -> bytes:
            checksum = binascii.crc32(kind + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

        forged = (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", 100_000, 100_000, 8, 2, 0, 0, 0),
            )
            + png_chunk(b"IEND", b"")
        )
        record = {
            "media_type": "image/png",
            "base64": base64.b64encode(forged).decode("ascii"),
        }
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "supported, valid image"):
                _decode_image_record(
                    record,
                    0,
                    Path(directory),
                    RequestLimits(max_image_pixels=20_000_000),
                )

    def test_encoded_snapshot_preserves_prompt_text_exactly(self) -> None:
        text = "00123\nNA,null,true,1e100"
        encoded, count = _encode_request({"text": text, "request_id": "0007"})
        self.assertEqual(count, 1)
        self.assertEqual(json.loads(encoded)["text"], text)
        self.assertEqual(json.loads(encoded)["request_id"], "0007")

    def test_non_string_prompt_is_rejected_instead_of_coerced(self) -> None:
        with self.assertRaisesRegex(GuardrailRequestError, "'text' must be a string"):
            _encode_request({"text": 123})

    def test_guarded_call_rejects_unevaluated_metadata_before_transport(self) -> None:
        client = GuardrailClient(
            "http://127.0.0.1:1",
            required_traffic_mode="enforce",
            timeout_seconds=0.1,
        )
        invoked = False

        def downstream(_request) -> None:
            nonlocal invoked
            invoked = True

        with self.assertRaisesRegex(GuardrailRequestError, "unevaluated fields"):
            client.guarded_call(
                {
                    "text": "benign carrier text",
                    "metadata": {"messages": ["unevaluated malicious instructions"]},
                },
                downstream,
            )
        self.assertFalse(invoked)

    def test_guarded_call_rejects_unevaluated_request_id(self) -> None:
        client = GuardrailClient(
            "http://127.0.0.1:1",
            required_traffic_mode="enforce",
            timeout_seconds=0.1,
        )
        with self.assertRaisesRegex(GuardrailRequestError, "request_id"):
            client.guarded_call(
                {
                    "text": "benign detector input",
                    "request_id": "unevaluated downstream instruction",
                },
                lambda _request: None,
            )

    def test_guarded_call_rejects_mutable_local_image_paths(self) -> None:
        client = GuardrailClient(
            "http://127.0.0.1:1",
            required_traffic_mode="enforce",
            timeout_seconds=0.1,
        )
        with self.assertRaisesRegex(GuardrailRequestError, "mutable local-path fields"):
            client.guarded_call(
                {"text": "prompt", "image_path": "replaceable.png"},
                lambda _request: None,
            )


class PromptMetadataTests(unittest.TestCase):
    def test_csv_staging_preserves_scalar_looking_text(self) -> None:
        values = [
            "00123",
            "NA",
            "null",
            "true",
            "1e100",
            "line 1\nline 2",
            "benign\x00 malicious instruction",
            "tab\tcarriage\rreturn",
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "text", "image_path"],
                )
                writer.writeheader()
                for index, value in enumerate(values):
                    writer.writerow(
                        {
                            "sample_id": f"{index:04d}",
                            "text": value,
                            "image_path": "",
                        }
                    )
            metadata = read_prompt_metadata(path)
        self.assertEqual(metadata["text"].tolist(), values)
        self.assertEqual(metadata["sample_id"].tolist()[0], "0000")
        self.assertEqual(metadata["image_path"].tolist(), [""] * len(values))


class ProxyPolicyTests(unittest.TestCase):
    def test_client_loopback_ignores_environment_proxy(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://proxy.invalid:8080",
                "http_proxy": "http://proxy.invalid:8080",
                "NO_PROXY": "",
                "no_proxy": "",
            },
            clear=False,
        ):
            proxy = environment_proxy_for_origin("http://127.0.0.1:8766")
        self.assertIsNone(proxy)

    def test_gui_loopback_ignores_environment_proxy(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://proxy.invalid:8080",
                "http_proxy": "http://proxy.invalid:8080",
                "NO_PROXY": "",
                "no_proxy": "",
            },
            clear=False,
        ):
            proxy = environment_proxy_for_origin("http://localhost:8766")
        self.assertIsNone(proxy)

    def test_remote_https_may_use_environment_connect_proxy(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "http://proxy.example:8443",
                "https_proxy": "http://proxy.example:8443",
                "NO_PROXY": "",
                "no_proxy": "",
            },
            clear=False,
        ):
            proxy = environment_proxy_for_origin("https://guard.example")
        self.assertEqual(proxy, "http://proxy.example:8443")


def _start_trickle_response_server(
    body: bytes,
    *,
    interval_seconds: float,
) -> tuple[str, socket.socket, Thread]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()

    def serve() -> None:
        connection = None
        try:
            connection, _ = listener.accept()
            request = bytearray()
            while b"\r\n\r\n" not in request:
                chunk = connection.recv(4096)
                if not chunk:
                    return
                request.extend(chunk)
            connection.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n".encode("ascii")
                + b"Connection: close\r\n\r\n"
            )
            for byte in body:
                connection.sendall(bytes((byte,)))
                time.sleep(interval_seconds)
        except OSError:
            pass
        finally:
            if connection is not None:
                connection.close()
            listener.close()

    thread = Thread(target=serve, daemon=True)
    thread.start()
    return f"http://{host}:{port}", listener, thread


def _start_raw_response_server(response: bytes) -> tuple[str, int, socket.socket, Thread]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()

    def serve() -> None:
        connection = None
        try:
            connection, _ = listener.accept()
            request = bytearray()
            while b"\r\n\r\n" not in request:
                chunk = connection.recv(4096)
                if not chunk:
                    return
                request.extend(chunk)
            connection.sendall(response)
        except OSError:
            pass
        finally:
            if connection is not None:
                connection.close()
            listener.close()

    thread = Thread(target=serve, daemon=True)
    thread.start()
    return host, port, listener, thread


class OutboundDeadlineTests(unittest.TestCase):
    def test_timeout_seconds_requires_a_finite_positive_number(self) -> None:
        invalid_values = (True, "1", float("nan"), float("inf"), 0, -1.0)
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "timeout_seconds must be positive and finite",
                ):
                    request_with_deadline(
                        "http://127.0.0.1:1",
                        "/",
                        method="GET",
                        headers={},
                        timeout_seconds=value,
                        max_response_bytes=16,
                    )

    def test_tls_context_creation_obeys_request_absolute_deadline(self) -> None:
        called = Event()

        def delayed_context_creation():
            called.set()
            time.sleep(0.35)
            return object()

        started = time.monotonic()
        with patch(
            "AEGIS.http_transport.ssl.create_default_context",
            side_effect=delayed_context_creation,
        ):
            with self.assertRaises(TimeoutError):
                request_with_deadline(
                    "https://127.0.0.1:1",
                    "/",
                    method="GET",
                    headers={},
                    timeout_seconds=0.05,
                    max_response_bytes=16,
                )
        self.assertTrue(called.is_set())
        self.assertLess(time.monotonic() - started, 0.2)

    def test_tls_context_error_does_not_expose_trust_store_details(self) -> None:
        sensitive_detail = r"C:\private\trust-store\client-secret.pem"
        with patch(
            "AEGIS.http_transport.ssl.create_default_context",
            side_effect=RuntimeError(sensitive_detail),
        ):
            with self.assertRaisesRegex(
                OSError,
                "TLS context initialization failed",
            ) as caught:
                request_with_deadline(
                    "https://127.0.0.1:1",
                    "/",
                    method="GET",
                    headers={},
                    timeout_seconds=1.0,
                    max_response_bytes=16,
                )
        self.assertNotIn(sensitive_detail, str(caught.exception))
        self.assertNotIn("client-secret", str(caught.exception))

    def test_proxy_bypass_obeys_request_absolute_deadline(self) -> None:
        called = Event()

        def delayed_proxy_bypass(_hostname: str) -> bool:
            called.set()
            time.sleep(0.35)
            return False

        started = time.monotonic()
        with (
            patch(
                "AEGIS.http_transport.ssl.create_default_context",
                return_value=object(),
            ),
            patch(
                "AEGIS.http_transport.proxy_bypass",
                side_effect=delayed_proxy_bypass,
            ),
        ):
            with self.assertRaises(TimeoutError):
                request_with_deadline(
                    "https://deadline.invalid",
                    "/",
                    method="GET",
                    headers={},
                    timeout_seconds=0.05,
                    max_response_bytes=16,
                )
        self.assertTrue(called.is_set())
        self.assertLess(time.monotonic() - started, 0.2)

    def test_getproxies_obeys_request_absolute_deadline(self) -> None:
        called = Event()

        def delayed_getproxies() -> dict[str, str]:
            called.set()
            time.sleep(0.35)
            return {"https": "http://proxy.invalid:8080"}

        started = time.monotonic()
        with (
            patch(
                "AEGIS.http_transport.ssl.create_default_context",
                return_value=object(),
            ),
            patch("AEGIS.http_transport.proxy_bypass", return_value=False),
            patch(
                "AEGIS.http_transport.getproxies",
                side_effect=delayed_getproxies,
            ),
        ):
            with self.assertRaises(TimeoutError):
                request_with_deadline(
                    "https://deadline.invalid",
                    "/",
                    method="GET",
                    headers={},
                    timeout_seconds=0.05,
                    max_response_bytes=16,
                )
        self.assertTrue(called.is_set())
        self.assertLess(time.monotonic() - started, 0.2)

    def test_proxy_discovery_error_does_not_expose_configuration(self) -> None:
        sensitive_url = "http://proxy-user:proxy-password@proxy.invalid:8080"
        with (
            patch(
                "AEGIS.http_transport.ssl.create_default_context",
                return_value=object(),
            ),
            patch(
                "AEGIS.http_transport.proxy_bypass",
                side_effect=RuntimeError(sensitive_url),
            ),
        ):
            with self.assertRaisesRegex(
                OSError,
                "Environment proxy discovery failed",
            ) as caught:
                request_with_deadline(
                    "https://guard.invalid",
                    "/",
                    method="GET",
                    headers={},
                    timeout_seconds=1.0,
                    max_response_bytes=16,
                )
        self.assertNotIn(sensitive_url, str(caught.exception))
        self.assertNotIn("proxy-password", str(caught.exception))

    def test_malformed_or_incomplete_http_is_normalized_as_transport_error(self) -> None:
        responses = (
            b"NOT-HTTP\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nabc",
        )
        for raw_response in responses:
            with self.subTest(response=raw_response[:20]):
                host, port, listener, thread = _start_raw_response_server(raw_response)
                try:
                    client = GuardrailClient(f"http://{host}:{port}", timeout_seconds=1)
                    with self.assertRaises(GuardrailTransportError):
                        client.guard({"text": "transport normalization"})
                finally:
                    listener.close()
                    thread.join(timeout=2)

    def test_dns_resolution_obeys_absolute_deadline(self) -> None:
        def delayed_resolution(*_args, **_kwargs):
            time.sleep(0.35)
            return []

        started = time.monotonic()
        with patch(
            "AEGIS.http_transport.socket.getaddrinfo",
            side_effect=delayed_resolution,
        ):
            with self.assertRaises(TimeoutError):
                request_with_deadline(
                    "http://deadline.invalid",
                    "/",
                    method="GET",
                    headers={},
                    timeout_seconds=0.05,
                    max_response_bytes=16,
                )
        self.assertLess(time.monotonic() - started, 0.2)

    def test_unusable_first_address_leaves_time_for_working_fallback(self) -> None:
        response = (
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
            b"Connection: close\r\n\r\n{}"
        )
        host, port, listener, thread = _start_raw_response_server(response)
        addresses = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.255.255.1", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", (host, port)),
        ]
        try:
            with patch("AEGIS.http_transport._resolve_host", return_value=addresses):
                result = request_with_deadline(
                    "http://dual-stack.test",
                    "/",
                    method="GET",
                    headers={},
                    timeout_seconds=0.6,
                    max_response_bytes=16,
                )
            self.assertEqual(result.status, 200)
            self.assertEqual(result.body, b"{}")
        finally:
            listener.close()
            thread.join(timeout=2)

    def test_client_deadline_bounds_trickled_response_body(self) -> None:
        origin, listener, thread = _start_trickle_response_server(
            b'{"ok":true}',
            interval_seconds=0.15,
        )
        client = GuardrailClient(origin, timeout_seconds=0.35)
        started = time.monotonic()
        try:
            with self.assertRaises(GuardrailTransportError):
                client.guard({"text": "deadline test"})
        finally:
            listener.close()
            thread.join(timeout=2.0)
        self.assertLess(time.monotonic() - started, 0.9)

    def test_gui_upstream_deadline_bounds_trickled_response_body(self) -> None:
        origin, listener, thread = _start_trickle_response_server(
            b'{"ok":true}',
            interval_seconds=0.15,
        )
        upstream = GuardrailUpstream(
            origin,
            api_token=None,
            admin_token=None,
            timeout_seconds=0.35,
        )
        started = time.monotonic()
        try:
            with self.assertRaises(UpstreamGuardrailError) as caught:
                upstream.readiness()
        finally:
            listener.close()
            thread.join(timeout=2.0)
        self.assertEqual(caught.exception.code, "service_unavailable")
        self.assertLess(time.monotonic() - started, 0.9)


class GUIServerConstructionTests(unittest.TestCase):
    _SESSION_TOKEN = "AegisGuiSession_7bJ4cN9mQ2xR6vK8pL5sT1wZ"

    def test_loopback_gui_server_constructs_with_retention_limits(self) -> None:
        with TemporaryDirectory() as directory:
            config = GUIServerConfig(
                history_path=Path(directory) / "gui-history.sqlite3",
                history_max_records=7,
                history_max_age_days=2.0,
                history_max_bytes=4096,
                project_root=Path.cwd(),
            )
            server = create_gui_http_server("127.0.0.1", 0, config)
            try:
                self.assertGreater(server.server_address[1], 0)
            finally:
                server.server_close()

    def test_gui_absolute_read_deadline_releases_trickled_header(self) -> None:
        with TemporaryDirectory() as directory:
            config = GUIServerConfig(
                history_path=Path(directory) / "gui-history.sqlite3",
                project_root=Path.cwd(),
                max_http_connections=1,
                request_read_timeout_seconds=0.5,
            )
            server = create_gui_http_server("127.0.0.1", 0, config)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address
            slow = socket.create_connection((host, port), timeout=2)
            slow.sendall(b"G")
            closed = False
            started = time.monotonic()
            while time.monotonic() - started < 0.9:
                time.sleep(0.1)
                try:
                    slow.sendall(b"E")
                    slow.settimeout(0.02)
                    data = slow.recv(4096)
                    if not data:
                        closed = True
                        break
                except socket.timeout:
                    pass
                except (BrokenPipeError, ConnectionResetError, OSError):
                    closed = True
                    break
            try:
                self.assertTrue(closed)
                connection = HTTPConnection(host, port, timeout=2)
                connection.request(
                    "GET",
                    "/",
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                connection.close()
            finally:
                slow.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_gui_session_header_gates_api_but_not_static_routes(self) -> None:
        with TemporaryDirectory() as directory:
            config = GUIServerConfig(
                history_path=Path(directory) / "gui-history.sqlite3",
                project_root=Path.cwd(),
                session_token=self._SESSION_TOKEN,
            )
            server = create_gui_http_server("127.0.0.1", 0, config)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address
            try:
                connection = HTTPConnection(host, port, timeout=2)
                connection.request("GET", "/")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIsNone(response.getheader("Set-Cookie"))
                self.assertEqual(response.getheader("Cache-Control"), "no-store")
                self.assertEqual(response.getheader("Referrer-Policy"), "no-referrer")
                response.read()
                connection.close()

                connection = HTTPConnection(host, port, timeout=2)
                connection.request(
                    "GET",
                    "/api/history",
                    headers={"Cookie": f"AEGIS_GUI_SESSION={self._SESSION_TOKEN}"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 401)
                self.assertEqual(
                    json.loads(response.read())["error"]["code"],
                    "gui_session_required",
                )
                self.assertIsNone(response.getheader("Set-Cookie"))
                connection.close()

                connection = HTTPConnection(host, port, timeout=2)
                connection.request(
                    "GET",
                    "/api/history",
                    headers={"X-AEGIS-GUI-Session": "wrong"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 401)
                response.read()
                connection.close()

                connection = HTTPConnection(host, port, timeout=2)
                connection.request(
                    "GET",
                    "/api/history?limit=1",
                    headers={"X-AEGIS-GUI-Session": self._SESSION_TOKEN},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["total"], 0)
                connection.close()
            finally:
                server.shutdown()
                server.begin_drain_and_wait()
                server.server_close()
                thread.join(timeout=2)

    def test_gui_rejects_nonstandard_json_constants(self) -> None:
        with TemporaryDirectory() as directory:
            config = GUIServerConfig(
                history_path=Path(directory) / "gui-history.sqlite3",
                project_root=Path.cwd(),
                session_token=self._SESSION_TOKEN,
            )
            server = create_gui_http_server("127.0.0.1", 0, config)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address
            session_header = self._SESSION_TOKEN
            origin = f"http://{host}:{port}"
            try:
                for method, path, body in (
                    ("POST", "/api/evaluate", b'{"text":NaN}'),
                    (
                        "PUT",
                        "/api/settings",
                        b'{"target_profile":"llava05b","traffic_mode":Infinity}',
                    ),
                ):
                    with self.subTest(method=method):
                        connection = HTTPConnection(host, port, timeout=2)
                        connection.request(
                            method,
                            path,
                            body=body,
                            headers={
                                "Content-Type": "application/json",
                                "Content-Length": str(len(body)),
                                "X-AEGIS-GUI-Session": session_header,
                                "Origin": origin,
                            },
                        )
                        response = connection.getresponse()
                        self.assertEqual(response.status, 400)
                        response.read()
                        connection.close()
            finally:
                server.shutdown()
                server.begin_drain_and_wait()
                server.server_close()
                thread.join(timeout=2)

    def test_shutdown_uses_independent_bearer_without_gui_session_header(self) -> None:
        api_token = "ApiBearer_4vN8xQ2mL7pK5sR9wT1zC6dH"
        shutdown_token = "ShutdownOnly_9pL4sT7wQ2xN6vK8mR5zC1dH"
        with TemporaryDirectory() as directory:
            config = GUIServerConfig(
                history_path=Path(directory) / "gui-history.sqlite3",
                project_root=Path.cwd(),
                api_token=api_token,
                session_token=self._SESSION_TOKEN,
                shutdown_token=shutdown_token,
            )
            server = create_gui_http_server("127.0.0.1", 0, config)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address

            connection = HTTPConnection(host, port, timeout=2)
            connection.request(
                "POST",
                "/api/shutdown",
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Length": "0",
                    "X-AEGIS-Confirmation": "shutdown-gui",
                },
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 401)
            response.read()
            connection.close()

            connection = HTTPConnection(host, port, timeout=2)
            connection.request(
                "POST",
                "/api/shutdown",
                headers={
                    "Authorization": f"Bearer {shutdown_token}",
                    "Content-Length": "0",
                    "X-AEGIS-Confirmation": "shutdown-gui",
                },
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 202)
            self.assertTrue(json.loads(response.read())["shutting_down"])
            connection.close()
            thread.join(timeout=3)
            server.begin_drain_and_wait()
            server.server_close()
            self.assertFalse(thread.is_alive())

    def test_gui_config_rejects_boolean_and_non_finite_timeouts(self) -> None:
        fields = (
            "upstream_timeout_seconds",
            "history_max_age_days",
            "target_switch_timeout_seconds",
            "request_read_timeout_seconds",
        )
        for field_name in fields:
            for invalid in (True, float("nan"), float("inf"), 0.0, -1.0):
                with self.subTest(field=field_name, value=invalid):
                    with self.assertRaisesRegex(ValueError, "positive and finite"):
                        GUIServerConfig(**{field_name: invalid})

    def test_target_controller_rejects_boolean_and_non_finite_timeouts(self) -> None:
        fields = (
            "startup_timeout_seconds",
            "poll_interval_seconds",
            "command_timeout_seconds",
        )
        for field_name in fields:
            for invalid in (True, float("nan"), float("inf"), 0.0, -1.0):
                with self.subTest(field=field_name, value=invalid):
                    with self.assertRaisesRegex(ValueError, "positive and finite"):
                        ComposeTargetController(
                            Path.cwd(),
                            object(),
                            command_runner=lambda *_args: None,
                            **{field_name: invalid},
                        )

    def test_cli_publishes_private_shutdown_state_and_drains_on_ctrl_c(self) -> None:
        from AEGIS.cli import _command_gui

        with TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "gui-state.json"
            observed: dict[str, object] = {}
            order: list[str] = []
            test_case = self

            class FakeServer:
                server_address = ("127.0.0.1", 8767)

                def serve_forever(self) -> None:
                    order.append("serve")
                    observed.update(json.loads(state_path.read_text(encoding="utf-8")))
                    if os.name != "nt":
                        self_mode = state_path.stat().st_mode & 0o777
                        test_case.assertEqual(self_mode, 0o600)
                    raise KeyboardInterrupt

                def begin_drain_and_wait(self) -> None:
                    order.append("drain")

                def server_close(self) -> None:
                    order.append("close")

            fake_server = FakeServer()
            args = SimpleNamespace(
                service_url="http://127.0.0.1:8766",
                host="127.0.0.1",
                port=8767,
                history_db=str(root / "gui-history.sqlite3"),
                state_file=str(state_path),
                history_max_records=100,
                history_max_age_days=30.0,
                history_max_bytes=1024 * 1024,
                api_token_env="AEGIS_TEST_MISSING_API_TOKEN",
                admin_token_env="AEGIS_TEST_MISSING_ADMIN_TOKEN",
                timeout_seconds=5.0,
                project_root=str(Path.cwd()),
                target_switch_timeout_seconds=5.0,
                max_http_connections=4,
                request_read_timeout_seconds=2.0,
                open_browser=True,
            )
            opened_urls: list[str] = []
            with (
                patch(
                    "AEGIS.gui_server.create_gui_http_server",
                    return_value=fake_server,
                ),
                patch("AEGIS.cli.signal.getsignal", return_value=signal.SIG_DFL),
                patch("AEGIS.cli.signal.signal"),
                patch("webbrowser.open_new_tab", side_effect=opened_urls.append),
                patch("builtins.print"),
            ):
                _command_gui(args)

            self.assertEqual(order, ["serve", "drain", "close"])
            self.assertEqual(observed["schema_version"], 1)
            self.assertEqual(observed["pid"], os.getpid())
            self.assertEqual(observed["host"], "127.0.0.1")
            self.assertEqual(observed["port"], 8767)
            shutdown_token = observed["shutdown_token"]
            self.assertIsInstance(shutdown_token, str)
            self.assertGreaterEqual(len(shutdown_token), 32)
            self.assertEqual(len(opened_urls), 1)
            self.assertIn("#session=", opened_urls[0])
            self.assertNotIn(str(shutdown_token), opened_urls[0])
            self.assertFalse(state_path.exists())

    def test_target_change_drains_active_evaluation_and_rejects_new_one(self) -> None:
        barrier = _HistoryActivityBarrier()
        evaluation_started = Event()
        release_evaluation = Event()
        target_started = Event()
        release_target = Event()

        def evaluate() -> None:
            with barrier.evaluation():
                evaluation_started.set()
                release_evaluation.wait(2)

        def change_target() -> None:
            with barrier.target_change():
                target_started.set()
                release_target.wait(2)

        evaluation_thread = Thread(target=evaluate)
        evaluation_thread.start()
        self.assertTrue(evaluation_started.wait(1))
        target_thread = Thread(target=change_target)
        target_thread.start()
        with barrier._condition:
            self.assertTrue(
                barrier._condition.wait_for(lambda: barrier._changing_target, timeout=1)
            )
        self.assertFalse(target_started.is_set())
        with self.assertRaises(GUITargetChangingError):
            with barrier.evaluation():
                self.fail("evaluation entered while a target change was pending")
        release_evaluation.set()
        self.assertTrue(target_started.wait(1))
        release_target.set()
        evaluation_thread.join(timeout=2)
        target_thread.join(timeout=2)
        self.assertFalse(evaluation_thread.is_alive())
        self.assertFalse(target_thread.is_alive())

    def test_settings_target_switch_waits_for_inflight_http_evaluation(self) -> None:
        evaluation_started = Event()
        release_evaluation = Event()
        status_checked = Event()
        switch_started = Event()
        release_switch = Event()

        class BlockingController:
            def begin_shutdown(self) -> None:
                return

            def wait_until_idle(self) -> None:
                release_switch.wait(2)

            def metadata(self, active_target):
                return {
                    "available": True,
                    "busy": switch_started.is_set() and not release_switch.is_set(),
                    "reason": None,
                    "operation": None,
                    "targets": [],
                }

            def switch(self, target_profile, previous_target, traffic_mode):
                switch_started.set()
                if not release_switch.wait(3):
                    raise AssertionError("test did not release the target switch")
                return {
                    "changed_target": True,
                    "active_target": target_profile,
                    "traffic_mode": traffic_mode,
                }

        def upstream_status(_self):
            status_checked.set()
            return {
                "connected": True,
                "ready": True,
                "error": None,
                "readiness": {
                    "ok": True,
                    "target_profile": "llava05b",
                    "traffic_mode": "shadow",
                },
                "capabilities": {
                    "input_modalities": ["text"],
                    "runtime_traffic_mode_control": True,
                },
                "metrics": {},
                "metrics_error": None,
            }

        def upstream_evaluate(_self, payload):
            evaluation_started.set()
            if not release_evaluation.wait(3):
                raise AssertionError("test did not release the evaluation")
            return {
                "trace_id": payload["request_id"],
                "decisions": [
                    {
                        "action": "allow",
                        "recommended_action": "allow",
                        "verdict": "benign",
                        "risk_score": 0.01,
                        "modality": "text",
                    }
                ],
                "summary": {"traffic_mode": "shadow"},
            }

        with TemporaryDirectory() as directory:
            config = GUIServerConfig(
                history_path=Path(directory) / "gui-history.sqlite3",
                project_root=Path.cwd(),
                api_token="ApiBearer_4vN8xQ2mL7pK5sR9wT1zC6dH",
                admin_token="AdminBearer_8mQ2vN5xL7pK4sR9wT1zC6dH",
                session_token=self._SESSION_TOKEN,
            )
            controller = BlockingController()
            with (
                patch.object(GuardrailUpstream, "status", upstream_status),
                patch.object(
                    GuardrailUpstream,
                    "input_capabilities",
                    lambda _self: (("text",), "llava05b"),
                ),
                patch.object(GuardrailUpstream, "evaluate", upstream_evaluate),
            ):
                server = create_gui_http_server(
                    "127.0.0.1",
                    0,
                    config,
                    target_controller=controller,
                )
                thread = Thread(target=server.serve_forever, daemon=True)
                thread.start()
                host, port = server.server_address
                origin = f"http://{host}:{port}"
                session_header = self._SESSION_TOKEN
                results: dict[str, int] = {}

                def post_evaluation(name: str, text: str) -> None:
                    connection = HTTPConnection(host, port, timeout=4)
                    body = json.dumps({"text": text}).encode("utf-8")
                    connection.request(
                        "POST",
                        "/api/evaluate",
                        body=body,
                        headers={
                            "Content-Type": "application/json",
                            "Content-Length": str(len(body)),
                            "X-AEGIS-GUI-Session": session_header,
                            "Origin": origin,
                        },
                    )
                    response = connection.getresponse()
                    results[name] = response.status
                    response.read()
                    connection.close()

                def put_settings() -> None:
                    connection = HTTPConnection(host, port, timeout=4)
                    body = json.dumps(
                        {
                            "target_profile": "qwen25vl3b",
                            "traffic_mode": "shadow",
                        }
                    ).encode("utf-8")
                    connection.request(
                        "PUT",
                        "/api/settings",
                        body=body,
                        headers={
                            "Content-Type": "application/json",
                            "Content-Length": str(len(body)),
                            "X-AEGIS-GUI-Session": session_header,
                            "Origin": origin,
                        },
                    )
                    response = connection.getresponse()
                    results["settings"] = response.status
                    response.read()
                    connection.close()

                evaluation_thread = Thread(
                    target=post_evaluation,
                    args=("first_evaluation", "first"),
                )
                settings_thread = Thread(target=put_settings)
                evaluation_thread.start()
                self.assertTrue(evaluation_started.wait(1))
                settings_thread.start()
                self.assertTrue(status_checked.wait(1))
                self.assertFalse(switch_started.is_set())

                release_evaluation.set()
                self.assertTrue(switch_started.wait(2))
                post_evaluation("second_evaluation", "second")
                self.assertEqual(results["second_evaluation"], 503)
                release_switch.set()

                evaluation_thread.join(timeout=3)
                settings_thread.join(timeout=3)
                self.assertEqual(results["first_evaluation"], 200)
                self.assertEqual(results["settings"], 200)
                server.shutdown()
                server.begin_drain_and_wait()
                server.server_close()
                thread.join(timeout=2)

    def test_runtime_control_requires_admin_token_in_gui_process(self) -> None:
        status = {
            "capabilities": {"runtime_traffic_mode_control": True},
            "metrics": {},
            "metrics_error": None,
        }
        self.assertFalse(
            _authenticated_runtime_control(status, admin_token_configured=False)
        )
        self.assertTrue(
            _authenticated_runtime_control(status, admin_token_configured=True)
        )

    def test_admin_preflight_failure_does_not_stop_active_target(self) -> None:
        class RejectingAdminUpstream:
            service_url = "http://127.0.0.1:8766"

            def readiness(self):
                return {
                    "ok": True,
                    "target_profile": "llava05b",
                    "traffic_mode": "shadow",
                }

            def set_traffic_mode(self, _traffic_mode, *, expected_target):
                self.expected_target = expected_target
                raise RuntimeError("admin credential rejected")

        commands = []

        def command_runner(arguments, _root, _timeout):
            commands.append(tuple(arguments))
            raise AssertionError("Compose must not run before admin authentication")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "compose.yaml").write_text(
                "services:\n  aegis-llava:\n  aegis-qwen:\n",
                encoding="utf-8",
            )
            controller = ComposeTargetController(
                root,
                RejectingAdminUpstream(),
                command_runner=command_runner,
            )
            with self.assertRaisesRegex(RuntimeError, "admin credential rejected"):
                controller.switch("qwen25vl3b", "llava05b", "shadow")
        self.assertEqual(commands, [])


class _FakeService:
    def __init__(self) -> None:
        self.api_token = "guard-token"
        self.admin_token = "admin-token"
        self.max_body_bytes = 4096
        self.metrics = ServiceMetrics()
        self.audit_logger = None
        self.target_profile = "test"
        self._traffic_mode = "enforce"

    def authorize(self, authorization: str | None) -> bool:
        return authorization is not None and hmac.compare_digest(
            authorization,
            f"Bearer {self.api_token}",
        )

    def authorize_admin(self, authorization: str | None) -> bool:
        return authorization is not None and hmac.compare_digest(
            authorization,
            f"Bearer {self.admin_token}",
        )

    def readiness_report(self) -> dict[str, object]:
        return {"ok": True}

    def current_traffic_mode(self) -> str:
        return self._traffic_mode

    def change_traffic_mode(self, requested: str) -> tuple[str, str]:
        previous = self._traffic_mode
        self._traffic_mode = requested
        return previous, requested

    def evaluate(self, _payload) -> dict[str, object]:
        return {
            "decisions": [],
            "summary": {
                "traffic_mode": self._traffic_mode,
                "action_counts": {},
            },
        }


class HTTPServerHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = _FakeService()
        self.server = create_guardrail_http_server(
            "127.0.0.1",
            0,
            self.service,
            max_http_connections=1,
            request_read_timeout_seconds=1.0,
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _put_mode(self, token: str) -> tuple[int, dict[str, object]]:
        connection = HTTPConnection(self.host, self.port, timeout=2)
        body = b'{"traffic_mode":"review"}'
        connection.request(
            "PUT",
            "/v1/admin/traffic-mode",
            body=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        status = response.status
        connection.close()
        self.assertTrue(self.server.wait_for_request_threads(2.0))
        return status, payload

    @staticmethod
    def _trickle_until_server_closes(
        connection: socket.socket,
        byte: bytes,
        *,
        maximum_seconds: float = 1.5,
    ) -> tuple[bool, bytes, float]:
        started = time.monotonic()
        response = bytearray()
        closed = False
        while time.monotonic() - started < maximum_seconds:
            time.sleep(0.15)
            try:
                connection.sendall(byte)
            except OSError:
                closed = True
                break
            connection.settimeout(0.03)
            try:
                chunk = connection.recv(4096)
                if not chunk:
                    closed = True
                    break
                response.extend(chunk)
                if b"408 Request Timeout" in response:
                    closed = True
                    break
            except socket.timeout:
                pass
            except OSError:
                closed = True
                break
        return closed, bytes(response), time.monotonic() - started

    def test_guard_token_cannot_authorize_admin_endpoint(self) -> None:
        status, payload = self._put_mode("guard-token")
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["code"], "unauthorized")
        status, payload = self._put_mode("admin-token")
        self.assertEqual(status, 200)
        self.assertEqual(payload["traffic_mode"], "review")

    def test_public_readiness_omits_diagnostics_and_status_requires_auth(self) -> None:
        detailed = {
            "ok": False,
            "target_profile": "llava05b",
            "traffic_mode": "review",
            "capabilities": {
                "input_modalities": ["text", "image", "unexpected"],
                "traffic_modes": ["shadow", "review", "enforce", "invalid"],
                "runtime_traffic_mode_control": True,
                "internal_note": "must-not-leak",
            },
            "model_revision": "private-model-revision",
            "audit_path": "C:/private/audit/session.jsonl",
            "worker_error": "private worker traceback",
        }
        self.service.readiness_report = lambda: detailed

        def get(path: str, token: str | None = None) -> tuple[int, dict[str, object]]:
            connection = HTTPConnection(self.host, self.port, timeout=2)
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            try:
                connection.request("GET", path, headers=headers)
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()
                self.assertTrue(self.server.wait_for_request_threads(2.0))

        status, public = get("/readyz")
        self.assertEqual(status, 503)
        self.assertEqual(
            public,
            {
                "ok": False,
                "target_profile": "llava05b",
                "traffic_mode": "review",
                "capabilities": {
                    "input_modalities": ["text", "image"],
                    "traffic_modes": ["shadow", "review", "enforce"],
                    "runtime_traffic_mode_control": True,
                },
                "reason": "service_not_ready",
            },
        )

        status, unauthorized = get("/v1/status")
        self.assertEqual(status, 401)
        self.assertEqual(unauthorized["error"]["code"], "unauthorized")

        status, authenticated = get("/v1/status", "guard-token")
        self.assertEqual(status, 503)
        self.assertEqual(authenticated, detailed)

    def test_connection_bound_and_read_timeout_release_capacity(self) -> None:
        slow = socket.create_connection((self.host, self.port), timeout=2)
        slow.sendall(
            b"POST /v1/guard HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Authorization: Bearer guard-token\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 100\r\n\r\n{"
        )
        time.sleep(0.1)

        overloaded = HTTPConnection(self.host, self.port, timeout=1)
        with self.assertRaises(
            (
                BrokenPipeError,
                ConnectionAbortedError,
                ConnectionResetError,
                RemoteDisconnected,
                TimeoutError,
            )
        ):
            overloaded.request("GET", "/livez")
            overloaded.getresponse()
        overloaded.close()

        slow.settimeout(2)
        response = slow.recv(4096)
        slow.close()
        self.assertIn(b"408 Request Timeout", response)

        connection = HTTPConnection(self.host, self.port, timeout=2)
        connection.request("GET", "/livez")
        healthy = connection.getresponse()
        self.assertEqual(healthy.status, 200)
        healthy.read()
        connection.close()

    def test_absolute_read_deadline_defeats_trickle_client(self) -> None:
        slow = socket.create_connection((self.host, self.port), timeout=2)
        slow.sendall(
            b"POST /v1/guard HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Authorization: Bearer guard-token\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 100\r\n\r\n{"
        )
        closed, response, elapsed = self._trickle_until_server_closes(slow, b" ")
        self.assertTrue(closed)
        self.assertLess(elapsed, 1.4)
        self.assertTrue(not response or b"408 Request Timeout" in response)

        connection = HTTPConnection(self.host, self.port, timeout=2)
        connection.request("GET", "/livez")
        healthy = connection.getresponse()
        self.assertEqual(healthy.status, 200)
        healthy.read()
        connection.close()
        slow.close()

    def test_absolute_read_deadline_releases_slow_header_connection(self) -> None:
        slow = socket.create_connection((self.host, self.port), timeout=2)
        slow.sendall(b"P")
        closed, _, elapsed = self._trickle_until_server_closes(slow, b"O")
        self.assertTrue(closed)
        self.assertLess(elapsed, 1.4)

        connection = HTTPConnection(self.host, self.port, timeout=2)
        connection.request("GET", "/livez")
        healthy = connection.getresponse()
        self.assertEqual(healthy.status, 200)
        healthy.read()
        connection.close()
        slow.close()

    def test_timer_start_failure_releases_connection_capacity(self) -> None:
        left, right = socket.socketpair()
        try:
            with patch(
                "AEGIS.http_server.Timer.start",
                side_effect=RuntimeError("thread creation failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "thread creation failed"):
                    self.server.process_request(left, ("local", 0))
            self.assertEqual(self.server._read_deadline_timers, {})
            self.assertTrue(self.server._connection_capacity.acquire(blocking=False))
            self.server._connection_capacity.release()
        finally:
            left.close()
            right.close()

    def test_server_close_is_bounded_while_accepted_inference_drains(self) -> None:
        evaluation_started = Event()
        release_evaluation = Event()
        request_finished = Event()

        def blocking_evaluate(_payload) -> dict[str, object]:
            evaluation_started.set()
            release_evaluation.wait(timeout=5.0)
            return {
                "decisions": [],
                "summary": {"traffic_mode": "enforce", "action_counts": {}},
            }

        self.service.evaluate = blocking_evaluate

        def request() -> None:
            connection = HTTPConnection(self.host, self.port, timeout=5)
            body = b'{"text":"bounded shutdown"}'
            try:
                connection.request(
                    "POST",
                    "/v1/guard",
                    body=body,
                    headers={
                        "Authorization": "Bearer guard-token",
                        "Content-Type": "application/json",
                        "Content-Length": str(len(body)),
                    },
                )
                response = connection.getresponse()
                response.read()
            finally:
                connection.close()
                request_finished.set()

        requester = Thread(target=request, daemon=True)
        requester.start()
        self.assertTrue(evaluation_started.wait(timeout=2.0))

        started = time.monotonic()
        self.server.shutdown()
        self.server.server_close()
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertFalse(self.server.wait_for_request_threads(0.1))

        release_evaluation.set()
        self.assertTrue(self.server.wait_for_request_threads(2.0))
        self.assertTrue(request_finished.wait(timeout=2.0))
        requester.join(timeout=2.0)

    def test_unsafe_request_id_is_not_reflected_in_response_header(self) -> None:
        connection = socket.create_connection((self.host, self.port), timeout=2)
        body = b"{}"
        connection.sendall(
            b"POST /v1/guard HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Authorization: Bearer guard-token\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 2\r\n"
            b"X-Request-ID: unsafe\x00identifier\r\n\r\n" + body
        )
        response = bytearray()
        connection.settimeout(2)
        while b"\r\n\r\n" not in response:
            response.extend(connection.recv(4096))
        connection.close()
        headers = bytes(response).split(b"\r\n\r\n", 1)[0]
        self.assertNotIn(b"unsafe\x00identifier", headers)
        self.assertRegex(headers, rb"X-Request-ID: [0-9a-f-]{36}")

    def test_ipv6_loopback_uses_ipv6_server_when_available(self) -> None:
        if not socket.has_ipv6:
            self.skipTest("Python reports no IPv6 support")
        try:
            server = create_guardrail_http_server(
                "::1",
                0,
                self.service,
                max_http_connections=1,
                request_read_timeout_seconds=1.0,
            )
        except OSError as exc:
            self.skipTest(f"IPv6 loopback is unavailable in this environment: {exc}")
        try:
            self.assertEqual(server.address_family, socket.AF_INET6)
        finally:
            server.server_close()

    def test_server_factory_rejects_unauthenticated_non_loopback_bind(self) -> None:
        service = _FakeService()
        service.api_token = None
        with self.assertRaisesRegex(ValueError, "Non-loopback API token is missing"):
            create_guardrail_http_server(
                "0.0.0.0",
                0,
                service,
                max_http_connections=1,
                request_read_timeout_seconds=1.0,
            )


if __name__ == "__main__":
    unittest.main()
