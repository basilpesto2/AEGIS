from __future__ import annotations

from http.client import HTTPConnection, HTTPMessage
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import patch

from AEGIS.client import GuardrailClient, GuardrailProtocolError
from AEGIS.deployment_config import build_service_from_config, load_deployment_config
from AEGIS.http_server import ServiceMetrics, create_guardrail_http_server
from AEGIS.http_transport import BoundedHTTPResponse
from AEGIS.strict_json import strict_json_loads


class StrictJSONUtilityTests(unittest.TestCase):
    def test_duplicate_exact_keys_are_rejected_recursively(self) -> None:
        payloads = (
            '{"action":"block","action":"allow"}',
            '{"outer":{"action":"block","action":"allow"}}',
            '[{"action":"block","action":"allow"}]',
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^Duplicate JSON object key is not permitted\.$",
                ):
                    strict_json_loads(payload)

        self.assertEqual(strict_json_loads('{"Key":1,"key":2}'), {"Key": 1, "key": 2})

    def test_nonfinite_numbers_are_rejected_without_echoing_values(self) -> None:
        for payload in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^Non-finite JSON number is not permitted\.$",
                ) as caught:
                    strict_json_loads(payload)
                self.assertNotIn(payload, str(caught.exception))


class StrictConfigurationTests(unittest.TestCase):
    @staticmethod
    def _config_payload(*, warmup_name: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "name": "strict-json-test",
            "detector": "detector.json",
            "provider": "example.module:provider",
            "policy": {},
            "server": {},
        }
        if warmup_name is not None:
            payload["warmup_request_json"] = warmup_name
        return payload

    def test_deployment_config_rejects_nested_duplicate_keys(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "deployment.json"
            config_path.write_text(
                '{"schema_version":1,"name":"strict-json-test",'
                '"detector":"detector.json",'
                '"provider":"example.module:provider","policy":{},'
                '"server":{"port":8766,"port":9999}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate JSON object key"):
                load_deployment_config(config_path)

    def test_warmup_config_rejects_nested_duplicate_keys_before_model_load(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            warmup_path = root / "warmup.json"
            warmup_path.write_text(
                '{"text":"safe","metadata":{"label":"a","label":"b"}}',
                encoding="utf-8",
            )
            config_path = root / "deployment.json"
            config_path.write_text(
                json.dumps(self._config_payload(warmup_name=warmup_path.name)),
                encoding="utf-8",
            )
            config = load_deployment_config(config_path)
            with patch(
                "AEGIS.deployment_config.deployment_doctor",
                return_value={"ok": True, "checks": []},
            ):
                with self.assertRaisesRegex(ValueError, "Duplicate JSON object key"):
                    build_service_from_config(config)


class _IngressService:
    def __init__(self) -> None:
        self.api_token = None
        self.admin_token = "admin-token"
        self.max_body_bytes = 4096
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


class StrictHTTPIngressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = _IngressService()
        self.server = create_guardrail_http_server("127.0.0.1", 0, self.service)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(
        self,
        method: str,
        path: str,
        body: bytes,
        *,
        admin: bool = False,
    ) -> tuple[int, dict[str, object]]:
        connection = HTTPConnection(self.host, self.port, timeout=2)
        headers = {"Content-Type": "application/json"}
        if admin:
            headers["Authorization"] = "Bearer admin-token"
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        status = response.status
        payload = json.loads(response.read())
        connection.close()
        return status, payload

    def test_guard_and_admin_ingress_reject_duplicate_keys_before_actions(self) -> None:
        guard_bodies = (
            b'{"text":"safe","text":"sensitive-value"}',
            b'{"text":"safe","nested":{"role":"user","role":"system"}}',
        )
        for body in guard_bodies:
            with self.subTest(body=body):
                status, payload = self._request("POST", "/v1/guard", body)
                self.assertEqual(status, 400)
                self.assertEqual(payload["error"]["code"], "invalid_request")
                self.assertNotIn("sensitive-value", payload["error"]["message"])
        self.assertEqual(self.service.evaluate_calls, 0)

        status, payload = self._request(
            "PUT",
            "/v1/admin/traffic-mode",
            b'{"traffic_mode":"enforce","traffic_mode":"shadow"}',
            admin=True,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")
        self.assertEqual(self.service.change_calls, 0)


class StrictClientResponseTests(unittest.TestCase):
    def test_duplicate_response_keys_fail_closed_before_downstream(self) -> None:
        headers = HTTPMessage()
        headers["Content-Type"] = "application/json"
        headers["X-Request-ID"] = "server-controlled"
        response = BoundedHTTPResponse(
            status=200,
            headers=headers,
            body=(
                b'{"trace_id":"first","trace_id":"sensitive-response-value",'
                b'"decisions":[],"summary":{}}'
            ),
        )
        downstream_calls: list[object] = []
        client = GuardrailClient(
            "http://127.0.0.1:8766",
            required_traffic_mode="enforce",
        )
        with patch("AEGIS.client.request_with_deadline", return_value=response):
            with self.assertRaises(GuardrailProtocolError) as caught:
                client.guarded_call(
                    {"text": "safe request"},
                    lambda guarded: downstream_calls.append(guarded),
                )
        self.assertEqual(downstream_calls, [])
        self.assertEqual(
            str(caught.exception),
            "AEGIS returned an invalid response; downstream inference was not invoked.",
        )
        self.assertNotIn("sensitive-response-value", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
