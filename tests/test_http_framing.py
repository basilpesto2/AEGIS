from __future__ import annotations

from email.message import Message
import json
import socket
from threading import Thread
import unittest

from AEGIS.http_framing import RequestFramingError, parse_request_content_length
from AEGIS.http_server import ServiceMetrics, create_guardrail_http_server


def _headers(*fields: tuple[str, str]) -> Message:
    headers = Message()
    for name, value in fields:
        headers[name] = value
    return headers


class RequestFramingUtilityTests(unittest.TestCase):
    def test_required_body_rejects_duplicate_content_length_even_if_identical(self) -> None:
        for values in (("2", "2"), ("2", "3")):
            with self.subTest(values=values):
                headers = _headers(
                    ("Content-Length", values[0]),
                    ("Content-Length", values[1]),
                )
                with self.assertRaisesRegex(
                    RequestFramingError,
                    "Multiple Content-Length fields",
                ) as caught:
                    parse_request_content_length(headers, body_required=True)
                self.assertEqual(caught.exception.reason, "duplicate_content_length")

    def test_required_body_accepts_only_one_positive_canonical_ascii_length(self) -> None:
        self.assertEqual(
            parse_request_content_length(
                _headers(("Content-Length", "2")),
                body_required=True,
            ),
            2,
        )
        invalid_values = ("", "0", "00", "02", "+2", "-2", " 2", "2 ", "2,2", "２")
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(RequestFramingError):
                    parse_request_content_length(
                        _headers(("Content-Length", value)),
                        body_required=True,
                    )

    def test_any_transfer_encoding_is_rejected_before_content_length(self) -> None:
        for headers in (
            _headers(("Transfer-Encoding", "chunked")),
            _headers(("Transfer-Encoding", ""), ("Content-Length", "2")),
            _headers(("Transfer-Encoding", "identity"), ("Content-Length", "2")),
        ):
            with self.subTest(headers=list(headers.items())):
                with self.assertRaises(RequestFramingError) as caught:
                    parse_request_content_length(headers, body_required=True)
                self.assertEqual(caught.exception.reason, "transfer_encoding")

    def test_bodyless_mode_accepts_only_absent_or_one_literal_zero(self) -> None:
        self.assertEqual(parse_request_content_length(Message(), body_required=False), 0)
        self.assertEqual(
            parse_request_content_length(
                _headers(("Content-Length", "0")),
                body_required=False,
            ),
            0,
        )
        rejected = (
            _headers(("Content-Length", "00")),
            _headers(("Content-Length", "1")),
            _headers(("Content-Length", "0"), ("Content-Length", "0")),
            _headers(("Transfer-Encoding", "chunked")),
        )
        for headers in rejected:
            with self.subTest(headers=list(headers.items())):
                with self.assertRaises(RequestFramingError):
                    parse_request_content_length(headers, body_required=False)


class _FramingService:
    def __init__(self) -> None:
        self.api_token = None
        self.admin_token = "admin-token"
        self.max_body_bytes = 4
        self.metrics = ServiceMetrics()
        self.audit_logger = None
        self.target_profile = "test"
        self.evaluate_calls = 0
        self.change_calls = 0
        self._traffic_mode = "enforce"

    def authorize(self, _authorization: str | None) -> bool:
        return True

    def authorize_admin(self, authorization: str | None) -> bool:
        return authorization == "Bearer admin-token"

    def readiness_report(self) -> dict[str, object]:
        return {"ok": True}

    def current_traffic_mode(self) -> str:
        return self._traffic_mode

    def change_traffic_mode(self, requested: str) -> tuple[str, str]:
        self.change_calls += 1
        previous = self._traffic_mode
        self._traffic_mode = requested
        return previous, requested

    def evaluate(self, _payload: object) -> dict[str, object]:
        self.evaluate_calls += 1
        return {
            "decisions": [],
            "summary": {"traffic_mode": self._traffic_mode, "action_counts": {}},
        }


class CoreHTTPFramingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = _FramingService()
        self.server = create_guardrail_http_server("127.0.0.1", 0, self.service)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _raw_request(
        self,
        method: str,
        path: str,
        header_lines: list[bytes],
        body: bytes = b"",
    ) -> tuple[int, dict[str, object]]:
        connection = socket.create_connection((self.host, self.port), timeout=2)
        with connection:
            connection.sendall(
                f"{method} {path} HTTP/1.1\r\n".encode("ascii")
                + b"Host: localhost\r\n"
                + b"Connection: close\r\n"
                + b"\r\n".join(header_lines)
                + b"\r\n\r\n"
                + body
            )
            connection.settimeout(2)
            response = bytearray()
            while True:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                response.extend(chunk)
        head, separator, encoded_body = bytes(response).partition(b"\r\n\r\n")
        self.assertEqual(separator, b"\r\n\r\n")
        status = int(head.split(b"\r\n", 1)[0].split()[1])
        return status, json.loads(encoded_body)

    def test_guard_rejects_same_and_different_duplicate_lengths_without_evaluation(self) -> None:
        for second_length in (b"2", b"3"):
            with self.subTest(second_length=second_length):
                status, payload = self._raw_request(
                    "POST",
                    "/v1/guard",
                    [
                        b"Content-Type: application/json",
                        b"Content-Length: 2",
                        b"Content-Length: " + second_length,
                    ],
                    b"{}",
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload["error"]["code"], "invalid_request")
        self.assertEqual(self.service.evaluate_calls, 0)

    def test_transfer_encoding_with_content_length_suppresses_admin_action(self) -> None:
        body = b'{"traffic_mode":"shadow"}'
        status, payload = self._raw_request(
            "PUT",
            "/v1/admin/traffic-mode",
            [
                b"Authorization: Bearer admin-token",
                b"Content-Type: application/json",
                b"Transfer-Encoding: chunked",
                f"Content-Length: {len(body)}".encode("ascii"),
            ],
            body,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")
        self.assertEqual(self.service.change_calls, 0)

    def test_existing_length_required_and_body_cap_semantics_are_preserved(self) -> None:
        status, payload = self._raw_request(
            "POST",
            "/v1/guard",
            [b"Content-Type: application/json", b"Content-Length: +2"],
            b"{}",
        )
        self.assertEqual(status, 411)
        self.assertEqual(payload["error"]["code"], "content_length_required")

        status, payload = self._raw_request(
            "POST",
            "/v1/guard",
            [b"Content-Type: application/json", b"Content-Length: 5"],
            b"12345",
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "request_too_large")
        self.assertEqual(self.service.evaluate_calls, 0)


if __name__ == "__main__":
    unittest.main()
