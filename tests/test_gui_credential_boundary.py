from __future__ import annotations

from contextlib import contextmanager
from http import HTTPStatus
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from AEGIS.gui_control import (
    CommandResult,
    ComposeTargetController,
    GUIControlError,
)
from AEGIS.gui_server import (
    GUIServerConfig,
    GuardrailUpstream,
    UpstreamGuardrailError,
    create_gui_http_server,
)
from AEGIS.target_profiles import get_target_profile


_LLAVA_ID = "a" * 64
_QWEN_ID = "b" * 64
_SERVICE_BY_TARGET = {
    "llava05b": "aegis-llava",
    "qwen25vl3b": "aegis-qwen",
}
_TARGET_BY_SERVICE = {value: key for key, value in _SERVICE_BY_TARGET.items()}


def _write_compose(root: Path) -> None:
    (root / "compose.yaml").write_text(
        "services:\n  aegis-llava:\n  aegis-qwen:\n",
        encoding="utf-8",
    )


def _raw_http_status(host: str, port: int, request: bytes) -> int:
    connection = socket.create_connection((host, port), timeout=2)
    try:
        connection.sendall(request)
        response = bytearray()
        connection.settimeout(2)
        while True:
            chunk = connection.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
    finally:
        connection.close()
    status_line = bytes(response).split(b"\r\n", 1)[0]
    return int(status_line.split(b" ", 2)[1])


class _ComposeHarness:
    def __init__(self, root: Path, active: str | None) -> None:
        self.root = root
        self.containers = {
            "llava05b": {"id": _LLAVA_ID, "running": active == "llava05b"},
            "qwen25vl3b": {"id": _QWEN_ID, "running": active == "qwen25vl3b"},
        }
        self.commands: list[tuple[str, ...]] = []
        self.exclusive_binding = True

    def run(self, arguments, _root, _timeout) -> CommandResult:
        argv = tuple(str(item) for item in arguments)
        self.commands.append(argv)
        if "compose" in argv and "ps" in argv:
            service = argv[-1]
            target = _TARGET_BY_SERVICE[service]
            state = self.containers[target]
            include = bool(state["running"]) or "--all" in argv
            return CommandResult(0, f"{state['id']}\n" if include else "")
        if len(argv) >= 2 and argv[1] == "inspect":
            container_id = argv[-1]
            target = self._target_for_id(container_id)
            return CommandResult(0, json.dumps([self._record(target)]))
        if "compose" in argv and "stop" in argv:
            target = _TARGET_BY_SERVICE[argv[-1]]
            self.containers[target]["running"] = False
            return CommandResult(0)
        if "compose" in argv and "up" in argv:
            target = _TARGET_BY_SERVICE[argv[-1]]
            self.containers[target]["running"] = True
            return CommandResult(0)
        if len(argv) >= 2 and argv[1] == "start":
            target = self._target_for_id(argv[-1])
            self.containers[target]["running"] = True
            return CommandResult(0, f"{argv[-1]}\n")
        if len(argv) >= 2 and argv[1] == "ps":
            identifiers = [
                str(state["id"])
                for state in self.containers.values()
                if state["running"]
            ]
            return CommandResult(0, "".join(f"{item}\n" for item in identifiers))
        raise AssertionError(f"Unexpected Docker command: {argv!r}")

    def _target_for_id(self, container_id: str) -> str:
        for target, state in self.containers.items():
            identifier = str(state["id"])
            if identifier.startswith(container_id) or container_id.startswith(identifier):
                return target
        raise AssertionError(f"Unknown container id {container_id!r}")

    def _record(self, target: str) -> dict[str, object]:
        state = self.containers[target]
        running = bool(state["running"])
        host_ip = "127.0.0.1" if self.exclusive_binding else "0.0.0.0"
        binding = [{"HostIp": host_ip, "HostPort": "8766"}]
        return {
            "Id": state["id"],
            "State": {"Running": running},
            "Config": {
                "Labels": {
                    "com.docker.compose.service": _SERVICE_BY_TARGET[target],
                    "com.docker.compose.oneoff": "False",
                    "com.docker.compose.project.working_dir": str(self.root),
                }
            },
            "HostConfig": {"PortBindings": {"8766/tcp": binding}},
            "NetworkSettings": {
                "Ports": {"8766/tcp": binding} if running else {}
            },
        }


class _RecordingUpstream:
    service_url = "http://127.0.0.1:8766"

    def __init__(
        self,
        harness: _ComposeHarness,
        *,
        fail_target: str | None = None,
    ) -> None:
        self.harness = harness
        self.fail_target = fail_target
        self.mode = "shadow"
        self.mode_targets: list[str] = []
        self.status_targets: list[str] = []
        self.include_detector_digest = True
        self.detector_digest: object | None = None

    def readiness(self):
        active = [
            target
            for target, state in self.harness.containers.items()
            if state["running"]
        ]
        target = active[0] if len(active) == 1 else None
        return {
            "ok": target is not None,
            "target_profile": target,
            "traffic_mode": self.mode,
        }

    def set_traffic_mode(self, traffic_mode, *, expected_target):
        self.mode_targets.append(expected_target)
        if expected_target == self.fail_target:
            raise RuntimeError("injected mode failure")
        self.mode = traffic_mode
        return {"traffic_mode": traffic_mode}

    def authenticated_status(self, *, expected_target):
        self.status_targets.append(expected_target)
        payload = {
            "ok": True,
            "target_profile": expected_target,
            "traffic_mode": self.mode,
        }
        if self.include_detector_digest:
            payload["detector_identity_sha256"] = (
                get_target_profile(expected_target).detector_identity_sha256
                if self.detector_digest is None
                else self.detector_digest
            )
        return payload


class GUIConfigBoundaryTests(unittest.TestCase):
    def test_config_repr_redacts_all_bearers(self) -> None:
        config = GUIServerConfig(
            api_token="A" * 32,
            admin_token="B" * 32,
            session_token="C" * 32,
            shutdown_token="D" * 32,
        )
        rendered = repr(config)
        for secret in ("A" * 32, "B" * 32, "C" * 32, "D" * 32):
            self.assertNotIn(secret, rendered)

    def test_integer_limits_require_exact_positive_ints(self) -> None:
        for field_name in (
            "history_max_records",
            "history_max_bytes",
            "max_http_connections",
        ):
            for invalid in (True, 1.0, "1", 0, -1):
                with self.subTest(field=field_name, value=invalid):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        GUIServerConfig(**{field_name: invalid})

    def test_timeout_and_age_values_reject_strings_bool_and_nonfinite(self) -> None:
        for field_name in (
            "upstream_timeout_seconds",
            "history_max_age_days",
            "target_switch_timeout_seconds",
            "request_read_timeout_seconds",
        ):
            for invalid in (True, "1", float("nan"), float("inf"), 0, -1):
                with self.subTest(field=field_name, value=invalid):
                    with self.assertRaisesRegex(ValueError, "positive and finite"):
                        GUIServerConfig(**{field_name: invalid})

    def test_controller_timeouts_reject_numeric_strings(self) -> None:
        for field_name in (
            "startup_timeout_seconds",
            "poll_interval_seconds",
            "command_timeout_seconds",
        ):
            with self.subTest(field=field_name):
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    ComposeTargetController(
                        Path.cwd(),
                        object(),
                        command_runner=lambda *_args: CommandResult(0),
                        **{field_name: "1"},
                    )

    def test_gui_bodies_reject_nested_duplicate_keys_and_float_overflow(self) -> None:
        session_token = "S" * 32
        with TemporaryDirectory() as directory:
            config = GUIServerConfig(
                history_path=Path(directory) / "history.sqlite3",
                project_root=Path.cwd(),
                session_token=session_token,
                shutdown_token="T" * 32,
            )
            server = create_gui_http_server("127.0.0.1", 0, config)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address
            origin = f"http://{host}:{port}"
            try:
                cases = (
                    (
                        "POST",
                        "/api/evaluate",
                        b'{"text":"test","metadata":{"source":"a","source":"b"}}',
                        "invalid_submission",
                    ),
                    (
                        "POST",
                        "/api/evaluate",
                        b'{"text":1e999}',
                        "invalid_submission",
                    ),
                    (
                        "PUT",
                        "/api/settings",
                        (
                            b'{"target_profile":"llava05b",'
                            b'"traffic_mode":"shadow","traffic_mode":"review"}'
                        ),
                        "invalid_settings",
                    ),
                )
                for method, path, body, error_code in cases:
                    with self.subTest(path=path, body=body):
                        connection = HTTPConnection(host, port, timeout=2)
                        connection.request(
                            method,
                            path,
                            body=body,
                            headers={
                                "Content-Type": "application/json",
                                "Content-Length": str(len(body)),
                                "X-AEGIS-GUI-Session": session_token,
                                "Origin": origin,
                            },
                        )
                        response = connection.getresponse()
                        self.assertEqual(response.status, HTTPStatus.BAD_REQUEST)
                        self.assertEqual(
                            json.loads(response.read())["error"]["code"],
                            error_code,
                        )
                        connection.close()
            finally:
                server.shutdown()
                server.begin_drain_and_wait()
                server.server_close()
                thread.join(timeout=2)

    def test_ambiguous_gui_framing_is_rejected_before_any_action(self) -> None:
        session_token = "S" * 32
        shutdown_token = "T" * 32
        actions = {"evaluate": 0, "settings": 0, "clear": 0}

        def capabilities(_upstream):
            actions["evaluate"] += 1
            return (("text",), "llava05b")

        def status(_upstream):
            actions["settings"] += 1
            return {}

        def clear(_store):
            actions["clear"] += 1
            return {"records": 0, "images": 0}

        with TemporaryDirectory() as directory:
            config = GUIServerConfig(
                history_path=Path(directory) / "history.sqlite3",
                project_root=Path.cwd(),
                session_token=session_token,
                shutdown_token=shutdown_token,
            )
            with (
                patch.object(GuardrailUpstream, "input_capabilities", capabilities),
                patch.object(GuardrailUpstream, "status", status),
                patch("AEGIS.gui_server.GUIHistoryStore.clear_all", clear),
            ):
                server = create_gui_http_server("127.0.0.1", 0, config)
                thread = Thread(target=server.serve_forever, daemon=True)
                thread.start()
                host, port = server.server_address
                host_header = f"{host}:{port}".encode("ascii")
                origin = f"http://{host}:{port}".encode("ascii")
                evaluation = b'{"text":"must not run"}'
                settings = (
                    b'{"target_profile":"llava05b",'
                    b'"traffic_mode":"shadow"}'
                )
                requests = (
                    (
                        b"POST /api/evaluate HTTP/1.1\r\n"
                        b"Host: " + host_header + b"\r\n"
                        b"Origin: " + origin + b"\r\n"
                        b"X-AEGIS-GUI-Session: "
                        + session_token.encode("ascii")
                        + b"\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(evaluation)}\r\n".encode("ascii")
                        + f"Content-Length: {len(evaluation)}\r\n".encode("ascii")
                        + b"Connection: close\r\n\r\n"
                        + evaluation
                    ),
                    (
                        b"PUT /api/settings HTTP/1.1\r\n"
                        b"Host: " + host_header + b"\r\n"
                        b"Origin: " + origin + b"\r\n"
                        b"X-AEGIS-GUI-Session: "
                        + session_token.encode("ascii")
                        + b"\r\nContent-Type: application/json\r\n"
                        b"Transfer-Encoding: chunked\r\n"
                        + f"Content-Length: {len(settings)}\r\n".encode("ascii")
                        + b"Connection: close\r\n\r\n"
                        + settings
                    ),
                    (
                        b"POST /api/shutdown HTTP/1.1\r\n"
                        b"Host: " + host_header + b"\r\n"
                        b"Authorization: Bearer "
                        + shutdown_token.encode("ascii")
                        + b"\r\nX-AEGIS-Confirmation: shutdown-gui\r\n"
                        b"Content-Length: 0\r\nContent-Length: 0\r\n"
                        b"Connection: close\r\n\r\n"
                    ),
                    (
                        b"DELETE /api/history HTTP/1.1\r\n"
                        b"Host: " + host_header + b"\r\n"
                        b"Origin: " + origin + b"\r\n"
                        b"X-AEGIS-GUI-Session: "
                        + session_token.encode("ascii")
                        + b"\r\nX-AEGIS-Confirmation: clear-history\r\n"
                        b"Transfer-Encoding: chunked\r\nContent-Length: 0\r\n"
                        b"Connection: close\r\n\r\n"
                    ),
                )
                try:
                    for request in requests:
                        with self.subTest(request=request.split(b"\r\n", 1)[0]):
                            self.assertEqual(
                                _raw_http_status(host, port, request),
                                HTTPStatus.BAD_REQUEST,
                            )
                    self.assertEqual(
                        actions,
                        {"evaluate": 0, "settings": 0, "clear": 0},
                    )
                    self.assertTrue(thread.is_alive())
                    connection = HTTPConnection(host, port, timeout=2)
                    connection.request("GET", "/")
                    response = connection.getresponse()
                    self.assertEqual(response.status, HTTPStatus.OK)
                    response.read()
                    connection.close()
                finally:
                    server.shutdown()
                    server.begin_drain_and_wait()
                    server.server_close()
                    thread.join(timeout=2)


class ComposeCredentialBoundaryTests(unittest.TestCase):
    def test_same_target_reuse_requires_matching_detector_digest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_compose(root)
            harness = _ComposeHarness(root, "llava05b")
            upstream = _RecordingUpstream(harness)
            controller = ComposeTargetController(
                root,
                upstream,
                command_runner=harness.run,
                sleep=lambda _seconds: None,
            )

            result = controller.switch("llava05b", "llava05b", "review")

            self.assertFalse(result["changed_target"])
            self.assertEqual(upstream.status_targets, ["llava05b"])
            self.assertEqual(upstream.mode_targets, ["llava05b", "llava05b"])
            self.assertFalse(
                any(
                    "stop" in command or "up" in command
                    for command in harness.commands
                )
            )

    def test_same_target_digest_mismatch_fails_before_mode_change(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_compose(root)
            harness = _ComposeHarness(root, "llava05b")
            upstream = _RecordingUpstream(harness)
            upstream.detector_digest = "f" * 64
            controller = ComposeTargetController(
                root,
                upstream,
                command_runner=harness.run,
                sleep=lambda _seconds: None,
            )

            with self.assertRaises(GUIControlError) as caught:
                controller.switch("llava05b", "llava05b", "review")

            self.assertEqual(caught.exception.code, "detector_profile_mismatch")
            self.assertIn("Restart AEGIS", str(caught.exception))
            self.assertIn("reload the dashboard", str(caught.exception))
            self.assertEqual(upstream.mode_targets, [])
            self.assertFalse(
                any(
                    "stop" in command or "up" in command
                    for command in harness.commands
                )
            )

    def test_same_target_missing_or_invalid_digest_fails_closed(self) -> None:
        cases = ((False, None), (True, "not-a-sha256"), (True, 123))
        for include_digest, digest in cases:
            with self.subTest(
                include_digest=include_digest,
                digest=digest,
            ), TemporaryDirectory() as directory:
                root = Path(directory)
                _write_compose(root)
                harness = _ComposeHarness(root, "llava05b")
                upstream = _RecordingUpstream(harness)
                upstream.include_detector_digest = include_digest
                upstream.detector_digest = digest
                controller = ComposeTargetController(
                    root,
                    upstream,
                    command_runner=harness.run,
                    sleep=lambda _seconds: None,
                )

                with self.assertRaises(GUIControlError) as caught:
                    controller.switch("llava05b", "llava05b", "review")

                self.assertEqual(caught.exception.code, "detector_profile_mismatch")
                self.assertEqual(upstream.mode_targets, [])

    def test_same_target_partial_mode_success_restores_authenticated_previous_mode(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_compose(root)
            harness = _ComposeHarness(root, "llava05b")

            class ReadinessFailsAfterChange(_RecordingUpstream):
                def __init__(self, compose_harness):
                    super().__init__(compose_harness)
                    self.readiness_calls = 0

                def readiness(self):
                    self.readiness_calls += 1
                    if self.readiness_calls == 1:
                        return super().readiness()
                    return {
                        "ok": False,
                        "target_profile": "llava05b",
                        "traffic_mode": self.mode,
                    }

            upstream = ReadinessFailsAfterChange(harness)
            clock = [0.0]

            def advance(seconds):
                clock[0] += seconds

            controller = ComposeTargetController(
                root,
                upstream,
                startup_timeout_seconds=1.0,
                poll_interval_seconds=1.0,
                command_runner=harness.run,
                sleep=advance,
                monotonic=lambda: clock[0],
            )

            with self.assertRaises(GUIControlError) as failed:
                controller.switch("llava05b", "llava05b", "review")

            self.assertEqual(failed.exception.code, "traffic_mode_change_failed")
            self.assertTrue(failed.exception.rollback_succeeded)
            self.assertIn("previous mode was restored", str(failed.exception))
            self.assertEqual(upstream.mode, "shadow")
            self.assertEqual(
                upstream.mode_targets,
                ["llava05b", "llava05b", "llava05b"],
            )
            self.assertEqual(upstream.status_targets, ["llava05b", "llava05b"])

    def test_same_target_rollback_reports_authenticated_verification_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_compose(root)
            harness = _ComposeHarness(root, "llava05b")

            class UnverifiableRollback(_RecordingUpstream):
                def __init__(self, compose_harness):
                    super().__init__(compose_harness)
                    self.readiness_calls = 0

                def readiness(self):
                    self.readiness_calls += 1
                    if self.readiness_calls == 1:
                        return super().readiness()
                    return {
                        "ok": False,
                        "target_profile": "llava05b",
                        "traffic_mode": self.mode,
                    }

                def authenticated_status(self, *, expected_target):
                    payload = super().authenticated_status(
                        expected_target=expected_target
                    )
                    if len(self.status_targets) > 1:
                        payload["traffic_mode"] = "review"
                    return payload

            upstream = UnverifiableRollback(harness)
            clock = [0.0]

            def advance(seconds):
                clock[0] += seconds

            controller = ComposeTargetController(
                root,
                upstream,
                startup_timeout_seconds=1.0,
                poll_interval_seconds=1.0,
                command_runner=harness.run,
                sleep=advance,
                monotonic=lambda: clock[0],
            )

            with self.assertRaises(GUIControlError) as failed:
                controller.switch("llava05b", "llava05b", "review")

            self.assertEqual(failed.exception.code, "traffic_mode_change_failed")
            self.assertFalse(failed.exception.rollback_succeeded)
            self.assertIn("Rollback failed", str(failed.exception))
            self.assertIn("operator verification", str(failed.exception))
            self.assertEqual(upstream.mode, "shadow")

    def test_owner_proof_requires_exact_expected_exclusive_compose_service(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_compose(root)
            harness = _ComposeHarness(root, "llava05b")
            upstream = _RecordingUpstream(harness)
            controller = ComposeTargetController(
                root,
                upstream,
                command_runner=harness.run,
            )

            with controller.credentialed_request("llava05b") as target:
                self.assertEqual(target, "llava05b")
            with self.assertRaises(GUIControlError) as mismatch:
                with controller.credentialed_request("qwen25vl3b"):
                    self.fail("mismatched target entered credential scope")
            self.assertEqual(mismatch.exception.code, "credential_owner_unverified")

            harness.exclusive_binding = False
            with self.assertRaises(GUIControlError) as broad_binding:
                with controller.credentialed_request("llava05b"):
                    self.fail("non-exclusive port binding entered credential scope")
            self.assertEqual(
                broad_binding.exception.code,
                "credential_owner_unverified",
            )

    def test_expected_target_is_propagated_for_all_switch_paths(self) -> None:
        scenarios = (
            (None, "qwen25vl3b", None, ["qwen25vl3b"], True),
            ("llava05b", "llava05b", None, ["llava05b", "llava05b"], True),
            (
                "llava05b",
                "qwen25vl3b",
                None,
                ["llava05b", "qwen25vl3b"],
                True,
            ),
            (
                "llava05b",
                "qwen25vl3b",
                "qwen25vl3b",
                ["llava05b", "qwen25vl3b", "llava05b"],
                False,
            ),
        )
        for previous, requested, fail_target, expected_calls, succeeds in scenarios:
            with self.subTest(
                previous=previous,
                requested=requested,
                fail_target=fail_target,
            ), TemporaryDirectory() as directory:
                root = Path(directory)
                _write_compose(root)
                harness = _ComposeHarness(root, previous)
                upstream = _RecordingUpstream(harness, fail_target=fail_target)
                controller = ComposeTargetController(
                    root,
                    upstream,
                    command_runner=harness.run,
                    sleep=lambda _seconds: None,
                )
                if succeeds:
                    controller.switch(requested, previous, "review")
                else:
                    with self.assertRaises(GUIControlError) as failed:
                        controller.switch(requested, previous, "review")
                    self.assertTrue(failed.exception.rollback_succeeded)
                self.assertEqual(upstream.mode_targets, expected_calls)

    def test_rollback_restarts_exact_retained_container_without_compose_up(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_compose(root)
            harness = _ComposeHarness(root, "llava05b")
            upstream = _RecordingUpstream(harness, fail_target="qwen25vl3b")
            controller = ComposeTargetController(
                root,
                upstream,
                command_runner=harness.run,
                sleep=lambda _seconds: None,
            )
            with self.assertRaises(GUIControlError) as failed:
                controller.switch("qwen25vl3b", "llava05b", "review")
            self.assertTrue(failed.exception.rollback_succeeded)
            docker_start = [
                argv
                for argv in harness.commands
                if len(argv) >= 2 and argv[1] == "start"
            ]
            self.assertEqual(docker_start, [("docker", "start", _LLAVA_ID)])
            previous_compose_up = [
                argv
                for argv in harness.commands
                if "compose" in argv
                and "up" in argv
                and argv[-1] == "aegis-llava"
            ]
            self.assertEqual(previous_compose_up, [])

    def test_partial_stop_failure_still_runs_rollback_transaction(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_compose(root)
            harness = _ComposeHarness(root, "llava05b")
            upstream = _RecordingUpstream(harness)
            original_run = harness.run
            failed_once = False

            def fail_after_stopping(arguments, cwd, timeout):
                nonlocal failed_once
                argv = tuple(str(item) for item in arguments)
                result = original_run(arguments, cwd, timeout)
                if "compose" in argv and "stop" in argv and not failed_once:
                    failed_once = True
                    return CommandResult(1, stderr="injected partial stop failure")
                return result

            controller = ComposeTargetController(
                root,
                upstream,
                command_runner=fail_after_stopping,
                sleep=lambda _seconds: None,
            )
            with self.assertRaises(GUIControlError) as failed:
                controller.switch("qwen25vl3b", "llava05b", "review")
            self.assertTrue(failed.exception.rollback_succeeded)
            self.assertTrue(harness.containers["llava05b"]["running"])
            self.assertFalse(harness.containers["qwen25vl3b"]["running"])

    def test_container_identity_loss_makes_rollback_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_compose(root)
            harness = _ComposeHarness(root, "llava05b")
            upstream = _RecordingUpstream(harness, fail_target="qwen25vl3b")
            original_run = harness.run

            def remove_previous_before_rollback(arguments, cwd, timeout):
                argv = tuple(str(item) for item in arguments)
                if (
                    "compose" in argv
                    and "ps" in argv
                    and "--all" in argv
                    and argv[-1] == "aegis-llava"
                ):
                    harness.commands.append(argv)
                    return CommandResult(0, "")
                return original_run(arguments, cwd, timeout)

            controller = ComposeTargetController(
                root,
                upstream,
                command_runner=remove_previous_before_rollback,
                sleep=lambda _seconds: None,
            )
            with self.assertRaises(GUIControlError) as failed:
                controller.switch("qwen25vl3b", "llava05b", "review")
            self.assertFalse(failed.exception.rollback_succeeded)
            self.assertFalse(
                any(
                    len(argv) >= 2 and argv[1] == "start"
                    for argv in harness.commands
                )
            )

    def test_replaced_container_id_is_rejected_before_mode_restore(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_compose(root)
            harness = _ComposeHarness(root, "llava05b")
            upstream = _RecordingUpstream(harness, fail_target="qwen25vl3b")
            original_run = harness.run

            def replace_after_start(arguments, cwd, timeout):
                argv = tuple(str(item) for item in arguments)
                result = original_run(arguments, cwd, timeout)
                if len(argv) >= 2 and argv[1] == "start":
                    harness.containers["llava05b"]["id"] = "c" * 64
                return result

            controller = ComposeTargetController(
                root,
                upstream,
                command_runner=replace_after_start,
                sleep=lambda _seconds: None,
            )
            with self.assertRaises(GUIControlError) as failed:
                controller.switch("qwen25vl3b", "llava05b", "review")
            self.assertFalse(failed.exception.rollback_succeeded)
            self.assertEqual(
                upstream.mode_targets,
                ["llava05b", "qwen25vl3b"],
            )

    def test_credential_scope_serializes_with_target_switch(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_compose(root)
            harness = _ComposeHarness(root, "llava05b")
            upstream = _RecordingUpstream(harness)
            controller = ComposeTargetController(
                root,
                upstream,
                command_runner=harness.run,
                sleep=lambda _seconds: None,
            )
            scope_entered = Event()
            release_scope = Event()
            switch_finished = Event()

            def hold_credential_scope() -> None:
                with controller.credentialed_request("llava05b"):
                    scope_entered.set()
                    release_scope.wait(2)

            def change_mode() -> None:
                controller.switch("llava05b", "llava05b", "review")
                switch_finished.set()

            holder = Thread(target=hold_credential_scope)
            switcher = Thread(target=change_mode)
            holder.start()
            self.assertTrue(scope_entered.wait(1))
            switcher.start()
            self.assertFalse(switch_finished.wait(0.1))
            self.assertEqual(upstream.mode_targets, [])
            release_scope.set()
            holder.join(timeout=2)
            switcher.join(timeout=2)
            self.assertFalse(holder.is_alive())
            self.assertFalse(switcher.is_alive())
            self.assertEqual(upstream.mode_targets, ["llava05b", "llava05b"])


class UpstreamCredentialBoundaryTests(unittest.TestCase):
    def test_authenticated_status_proves_owner_before_sending_bearer(self) -> None:
        upstream = GuardrailUpstream(
            "http://127.0.0.1:8766",
            api_token="api-secret",
            admin_token="admin-secret",
            timeout_seconds=1.0,
        )
        events: list[object] = []

        @contextmanager
        def owner_guard(expected_target):
            events.append(("owner", expected_target))
            yield expected_target

        def request(path, *, headers, **_kwargs):
            events.append(("request", path, headers.get("Authorization")))
            return (
                {
                    "ok": True,
                    "target_profile": "llava05b",
                    "detector_sha256": get_target_profile(
                        "llava05b"
                    ).detector_sha256,
                },
                {},
            )

        upstream.set_local_credential_guard(owner_guard)
        upstream._request_json = request  # type: ignore[method-assign]

        status = upstream.authenticated_status(expected_target="llava05b")

        self.assertTrue(status["ok"])
        self.assertEqual(
            events,
            [
                ("owner", "llava05b"),
                ("request", "/v1/status", "Bearer api-secret"),
            ],
        )

    def test_authenticated_status_without_owner_proof_sends_no_bearer(self) -> None:
        upstream = GuardrailUpstream(
            "http://127.0.0.1:8766",
            api_token="api-secret",
            admin_token="admin-secret",
            timeout_seconds=1.0,
        )
        requests: list[str] = []
        upstream._request_json = (  # type: ignore[method-assign]
            lambda path, **_kwargs: requests.append(path)
        )

        with self.assertRaises(UpstreamGuardrailError) as caught:
            upstream.authenticated_status(expected_target="llava05b")

        self.assertEqual(caught.exception.code, "credential_owner_unverified")
        self.assertEqual(requests, [])

    def test_upstream_json_rejects_nested_duplicates_and_float_overflow(self) -> None:
        class Headers(dict):
            def get_content_type(self):
                return "application/json"

        upstream = GuardrailUpstream(
            "https://aegis.example.test",
            api_token="api-secret",
            admin_token="admin-secret",
            timeout_seconds=1.0,
        )
        for body in (
            b'{"ok":true,"details":{"worker":1,"worker":2}}',
            b'{"risk":1e999}',
        ):
            with self.subTest(body=body), patch(
                "AEGIS.gui_server.request_with_deadline",
                return_value=SimpleNamespace(
                    status=HTTPStatus.OK,
                    headers=Headers(),
                    body=body,
                ),
            ):
                with self.assertRaises(UpstreamGuardrailError) as caught:
                    upstream._request_json(
                        "/v1/status",
                        method="GET",
                        headers={"Authorization": "Bearer api-secret"},
                        timeout=1.0,
                    )
                self.assertEqual(caught.exception.code, "invalid_upstream_response")

    def test_spoofed_loopback_listener_never_receives_api_or_admin_bearer(self) -> None:
        observed: list[tuple[str, str | None]] = []

        class SpoofHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                observed.append((self.path, self.headers.get("Authorization")))
                payload = {
                    "ok": True,
                    "target_profile": "llava05b",
                    "traffic_mode": "shadow",
                    "capabilities": {"input_modalities": ["text"]},
                }
                self._reply(payload)

            def do_POST(self):
                observed.append((self.path, self.headers.get("Authorization")))
                self._reply({})

            def do_PUT(self):
                observed.append((self.path, self.headers.get("Authorization")))
                self._reply({})

            def _reply(self, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        listener = ThreadingHTTPServer(("127.0.0.1", 0), SpoofHandler)
        listener_thread = Thread(target=listener.serve_forever, daemon=True)
        listener_thread.start()
        port = listener.server_address[1]
        try:
            with TemporaryDirectory() as directory, patch(
                "AEGIS.gui_control._GUARDRAIL_HOST_PORT",
                str(port),
            ):
                root = Path(directory)
                _write_compose(root)
                upstream = GuardrailUpstream(
                    f"http://127.0.0.1:{port}",
                    api_token="api-secret",
                    admin_token="admin-secret",
                    timeout_seconds=1.0,
                )
                controller = ComposeTargetController(
                    root,
                    upstream,
                    command_runner=lambda *_args: CommandResult(0, ""),
                )
                upstream.set_local_credential_guard(controller.credentialed_request)

                with self.assertRaises(UpstreamGuardrailError) as evaluation:
                    upstream.evaluate({"text": "spoof check"})
                self.assertEqual(
                    evaluation.exception.code,
                    "credential_owner_unverified",
                )
                status = upstream.status()
                self.assertEqual(
                    status["metrics_error"]["code"],
                    "credential_owner_unverified",
                )
                with self.assertRaises(UpstreamGuardrailError) as admin:
                    upstream.set_traffic_mode(
                        "shadow",
                        expected_target="llava05b",
                    )
                self.assertEqual(admin.exception.code, "credential_owner_unverified")

            self.assertEqual([path for path, _ in observed], ["/readyz", "/readyz"])
            self.assertTrue(all(authorization is None for _, authorization in observed))
        finally:
            listener.shutdown()
            listener.server_close()
            listener_thread.join(timeout=2)

    def test_status_uses_public_identity_then_authenticated_full_status(self) -> None:
        upstream = GuardrailUpstream(
            "https://aegis.example.test",
            api_token="api-secret",
            admin_token="admin-secret",
            timeout_seconds=1.0,
        )
        requests: list[tuple[str, str | None]] = []

        def request(path, *, headers, **_kwargs):
            requests.append((path, headers.get("Authorization")))
            if path == "/readyz":
                return (
                    {"ok": True, "target_profile": "llava05b"},
                    {},
                )
            if path == "/v1/status":
                return (
                    {
                        "ok": True,
                        "target_profile": "llava05b",
                        "traffic_mode": "shadow",
                        "capabilities": {"input_modalities": ["text"]},
                    },
                    {},
                )
            return ({}, {})

        upstream._request_json = request  # type: ignore[method-assign]
        status = upstream.status()
        self.assertTrue(status["ready"])
        self.assertEqual(
            requests,
            [
                ("/readyz", None),
                ("/v1/status", "Bearer api-secret"),
                ("/metrics", "Bearer api-secret"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
