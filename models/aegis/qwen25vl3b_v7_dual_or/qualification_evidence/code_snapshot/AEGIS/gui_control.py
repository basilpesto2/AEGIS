from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import ipaddress
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from threading import Lock, RLock
import time
from typing import Any, Callable, Iterator, Protocol, Sequence
from urllib.parse import urlsplit
import uuid

from AEGIS.audit import validate_traffic_mode
from AEGIS.provenance import is_sha256
from AEGIS.strict_json import strict_json_loads
from AEGIS.target_profiles import get_target_profile


@dataclass(frozen=True)
class ComposeTarget:
    target_profile: str
    compose_profile: str
    service_name: str


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class TargetControlUpstream(Protocol):
    @property
    def service_url(self) -> str: ...

    def readiness(self) -> dict[str, Any]: ...

    def authenticated_status(
        self,
        *,
        expected_target: str,
    ) -> dict[str, Any]: ...

    def set_traffic_mode(
        self,
        traffic_mode: str,
        *,
        expected_target: str,
    ) -> dict[str, Any]: ...


CommandRunner = Callable[[Sequence[str], Path, float], CommandResult]


class GUIControlError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "target_control_error",
        rollback_succeeded: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.rollback_succeeded = rollback_succeeded


class GUIControlBusyError(GUIControlError):
    def __init__(self) -> None:
        super().__init__(
            "Another target change is already in progress.",
            code="target_control_busy",
        )


_TARGETS = {
    "llava05b": ComposeTarget(
        target_profile="llava05b",
        compose_profile="llava",
        service_name="aegis-llava",
    ),
    "qwen25vl3b": ComposeTarget(
        target_profile="qwen25vl3b",
        compose_profile="qwen",
        service_name="aegis-qwen",
    ),
}

_CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{12,64}")
_GUARDRAIL_CONTAINER_PORT = "8766/tcp"
_GUARDRAIL_HOST = "127.0.0.1"
_GUARDRAIL_HOST_PORT = "8766"


class ComposeTargetController:
    """Serialize safe target changes for the bundled local Compose deployment.

    The GUI remains a host-side companion. It may start and stop only the fixed
    AEGIS service names above; browser input is never incorporated into command
    arguments. The supplied upstream object is used to confirm that the desired
    target is actually ready and to apply its requested traffic mode.
    """

    def __init__(
        self,
        project_root: str | Path,
        upstream: TargetControlUpstream,
        *,
        startup_timeout_seconds: float = 1200.0,
        poll_interval_seconds: float = 1.0,
        command_timeout_seconds: float = 1800.0,
        command_runner: CommandRunner | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        _require_positive_finite(startup_timeout_seconds, "startup_timeout_seconds")
        _require_positive_finite(poll_interval_seconds, "poll_interval_seconds")
        _require_positive_finite(command_timeout_seconds, "command_timeout_seconds")

        self.project_root = Path(project_root).expanduser().resolve()
        self.compose_file = self.project_root / "compose.yaml"
        self.upstream = upstream
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.command_timeout_seconds = float(command_timeout_seconds)
        self._sleep = sleep
        self._monotonic = monotonic
        self._command_runner = command_runner or _run_command
        self._using_default_runner = command_runner is None
        self._docker_executable = (
            _resolve_docker_executable() if self._using_default_runner else "docker"
        )
        self._operation_lock = Lock()
        self._credential_lock = RLock()
        self._state_lock = Lock()
        self._operation: dict[str, object] | None = None
        self._accepting_operations = True
        self._availability_reason = self._validate_environment()

    @property
    def available(self) -> bool:
        return self._availability_reason is None

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return self._operation is not None

    def wait_until_idle(self) -> None:
        """Wait for target changes and credentialed local requests to finish."""

        self._operation_lock.acquire()
        self._operation_lock.release()
        self._credential_lock.acquire()
        self._credential_lock.release()

    @contextmanager
    def credentialed_request(
        self,
        expected_target: str | None,
    ) -> Iterator[str]:
        """Serialize and prove the Compose owner before a local bearer is sent."""

        self._credential_lock.acquire()
        try:
            with self._state_lock:
                if not self._accepting_operations and self._operation is None:
                    raise GUIControlError(
                        "The dashboard is shutting down and cannot send credentials.",
                        code="shutting_down",
                    )
            yield self._assert_credential_owner(expected_target)
        finally:
            self._credential_lock.release()

    def begin_shutdown(self) -> None:
        """Reject future switches without waiting for a current one to finish."""

        with self._state_lock:
            self._accepting_operations = False

    def begin_shutdown_and_wait(self) -> None:
        """Reject future switches and wait for any current operation to finish."""

        self.begin_shutdown()
        self.wait_until_idle()

    def metadata(self, active_target: str | None) -> dict[str, object]:
        with self._state_lock:
            operation = (
                None if self._operation is None else dict(self._operation)
            )
        targets = []
        for name, target in _TARGETS.items():
            profile = get_target_profile(name)
            targets.append(
                {
                    "id": name,
                    "name": name,
                    "title": profile.title,
                    "description": profile.description,
                    "status": profile.status,
                    "model_family": profile.model_family,
                    "model_id": profile.provider_options.get("model_id"),
                    "detector_mode": profile.detector_mode,
                    "detector_identity_sha256": profile.detector_identity_sha256,
                    "input_modalities": list(profile.intended_modalities),
                    "caveats": list(profile.caveats),
                    "compose_profile": target.compose_profile,
                    "service_name": target.service_name,
                    "active": name == active_target,
                }
            )
        return {
            "available": self.available,
            "busy": operation is not None,
            "reason": self._availability_reason,
            "project_root": str(self.project_root),
            "active_target": active_target,
            "operation": operation,
            "targets": targets,
        }

    def switch(
        self,
        target_profile: str,
        previous_target: str | None,
        traffic_mode: str = "shadow",
    ) -> dict[str, object]:
        target = self._target(target_profile)
        previous = None if previous_target is None else self._target(previous_target)
        mode = validate_traffic_mode(traffic_mode)
        same_target = previous is not None and previous == target
        if not same_target and not self.available:
            raise GUIControlError(
                self._availability_reason
                or "Local Compose target control is unavailable.",
                code="target_control_unavailable",
            )
        with self._state_lock:
            if not self._accepting_operations:
                raise GUIControlError(
                    "The dashboard is shutting down and cannot change targets.",
                    code="shutting_down",
                )
            acquired = self._operation_lock.acquire(blocking=False)
        if not acquired:
            raise GUIControlBusyError()

        operation_id = str(uuid.uuid4())
        self._set_operation(
            operation_id=operation_id,
            stage="starting",
            target_profile=target.target_profile,
            previous_target=(
                None if previous is None else previous.target_profile
            ),
        )
        try:
            # Status/evaluation requests share this lock. It spans every ownership
            # proof, credentialed HTTP request, and Compose mutation so this
            # controller cannot swap the verified listener before a bearer is sent.
            with self._credential_lock:
                self._set_operation_stage("authenticating_runtime_control")
                previous_mode = self._authenticate_runtime_control(
                    previous,
                    require_detector_match=same_target,
                )
                if same_target:
                    try:
                        self._set_operation_stage("setting_traffic_mode")
                        mode_result = self.upstream.set_traffic_mode(
                            mode,
                            expected_target=target.target_profile,
                        )
                        readiness = self._wait_until_ready(target.target_profile)
                    except Exception as exc:
                        rollback_succeeded = False
                        try:
                            self._set_operation_stage("rolling_back")
                            self.upstream.set_traffic_mode(
                                previous_mode,
                                expected_target=target.target_profile,
                            )
                            rollback_succeeded = self._authenticated_mode_matches(
                                target.target_profile,
                                previous_mode,
                            )
                        except Exception:
                            rollback_succeeded = False
                        raise GUIControlError(
                            (
                                "The traffic-mode change could not be verified. "
                                + (
                                    "The previous mode was restored."
                                    if rollback_succeeded
                                    else (
                                        "Rollback failed: the authenticated service "
                                        "did not confirm the previous mode. The active "
                                        "mode may require operator verification."
                                    )
                                )
                            ),
                            code="traffic_mode_change_failed",
                            rollback_succeeded=rollback_succeeded,
                        ) from exc
                    return {
                        "operation_id": operation_id,
                        "changed_target": False,
                        "active_target": target.target_profile,
                        "traffic_mode": _reported_mode(mode_result, readiness, mode),
                        "rollback_succeeded": None,
                    }

                previous_container_id = None
                try:
                    if previous is not None:
                        self._set_operation_stage("recording_previous_identity")
                        _, previous_container_id = self._credential_owner_details(
                            previous.target_profile
                        )
                        self._set_operation_stage("stopping_previous")
                        self._compose(previous, "stop")
                    else:
                        self._set_operation_stage("stopping_conflicts")
                        for candidate in _TARGETS.values():
                            if candidate != target:
                                self._compose(candidate, "stop")

                    self._set_operation_stage("starting_target")
                    self._compose(target, "start")
                    self._set_operation_stage("waiting_for_readiness")
                    readiness = self._wait_until_ready(target.target_profile)
                    self._set_operation_stage("setting_traffic_mode")
                    mode_result = self.upstream.set_traffic_mode(
                        mode,
                        expected_target=target.target_profile,
                    )
                    readiness = self._wait_until_ready(target.target_profile)
                except Exception as exc:
                    rollback_succeeded = self._rollback(
                        target,
                        previous,
                        previous_mode,
                        previous_container_id,
                    )
                    if previous is None:
                        rollback_message = (
                            "The failed target was stopped; there was no previous "
                            "target to restore."
                            if rollback_succeeded
                            else (
                                "The failed target could not be stopped, and there "
                                "was no previous target to restore."
                            )
                        )
                    else:
                        rollback_message = (
                            "The previous target and traffic mode were restored."
                            if rollback_succeeded
                            else (
                                "The previous target and traffic mode could not both "
                                "be restored."
                            )
                        )
                    raise GUIControlError(
                        (
                            f"Target {target.target_profile!r} could not be activated. "
                            f"{rollback_message}"
                        ),
                        code="target_switch_failed",
                        rollback_succeeded=rollback_succeeded,
                    ) from exc

                return {
                    "operation_id": operation_id,
                    "changed_target": True,
                    "active_target": target.target_profile,
                    "traffic_mode": _reported_mode(mode_result, readiness, mode),
                    "rollback_succeeded": None,
                }
        finally:
            self._clear_operation()
            self._operation_lock.release()

    def _rollback(
        self,
        attempted: ComposeTarget,
        previous: ComposeTarget | None,
        previous_mode: str,
        previous_container_id: str | None,
    ) -> bool:
        self._set_operation_stage("rolling_back")
        cleanup_succeeded = True
        try:
            self._compose(attempted, "stop")
        except Exception:
            cleanup_succeeded = False
        if previous is None:
            return cleanup_succeeded
        try:
            if previous_container_id is None:
                return False
            self._start_existing(previous, previous_container_id)
            self._wait_until_ready(previous.target_profile)
            _, restored_container_id = self._credential_owner_details(
                previous.target_profile
            )
            if not _same_container_id(restored_container_id, previous_container_id):
                return False
            self.upstream.set_traffic_mode(
                previous_mode,
                expected_target=previous.target_profile,
            )
            self._wait_until_ready(previous.target_profile)
            _, final_container_id = self._credential_owner_details(
                previous.target_profile
            )
            if not _same_container_id(final_container_id, previous_container_id):
                return False
            return cleanup_succeeded
        except Exception:
            return False

    def _authenticate_runtime_control(
        self,
        previous: ComposeTarget | None,
        *,
        require_detector_match: bool,
    ) -> str:
        """Prove the admin credential before any Compose mutation.

        Reapplying the current mode is intentionally a no-op. It prevents a missing
        or stale GUI admin token from stopping a healthy target and discovering the
        authentication error only after the replacement model has started.
        """
        if previous is None:
            # With no claimed active target there is nothing to authenticate.
            # The requested fixed service is started first, then the owner guard
            # protects its initial admin request.
            return "shadow"
        readiness = self.upstream.readiness()
        if not bool(readiness.get("ok")):
            raise GUIControlError(
                "The active AEGIS service is not ready for runtime control.",
                code="runtime_control_unavailable",
            )
        active_target = readiness.get("target_profile")
        if active_target != previous.target_profile:
            raise GUIControlError(
                "The active AEGIS target changed before the settings operation began.",
                code="active_target_changed",
            )
        current_mode = validate_traffic_mode(readiness.get("traffic_mode"))
        if require_detector_match:
            # Prove the fixed Compose owner before the upstream is permitted to
            # send the API bearer used by /v1/status. GuardrailUpstream repeats
            # this proof inside its credential scope, closing the listener-swap
            # interval between this check and the HTTP request.
            self._assert_credential_owner(previous.target_profile)
            status = self.upstream.authenticated_status(
                expected_target=previous.target_profile,
            )
            expected_digest = get_target_profile(
                previous.target_profile
            ).detector_identity_sha256.lower()
            reported_digest = status.get("detector_identity_sha256")
            if reported_digest is None:
                reported_digest = status.get("detector_sha256")
            valid_digest = (
                isinstance(reported_digest, str)
                and is_sha256(reported_digest)
            )
            if (
                status.get("ok") is not True
                or status.get("target_profile") != previous.target_profile
                or not valid_digest
                or reported_digest.lower() != expected_digest
            ):
                raise GUIControlError(
                    (
                        "The running detector does not match the selected target "
                        "profile. Restart AEGIS to load the current detector, then "
                        "reload the dashboard."
                    ),
                    code="detector_profile_mismatch",
                )
        self.upstream.set_traffic_mode(
            current_mode,
            expected_target=previous.target_profile,
        )
        return current_mode

    def _authenticated_mode_matches(
        self,
        target_profile: str,
        expected_mode: str,
    ) -> bool:
        """Verify a restored mode without depending on model readiness.

        A traffic-mode update can succeed while a subsequent public readiness
        poll fails for an unrelated model-health reason. Rollback therefore uses
        the authenticated full-status endpoint, inside the fixed Compose-owner
        credential boundary, as the authoritative mode observation.
        """

        self._assert_credential_owner(target_profile)
        status = self.upstream.authenticated_status(
            expected_target=target_profile,
        )
        if status.get("target_profile") != target_profile:
            return False
        try:
            reported_mode = validate_traffic_mode(status.get("traffic_mode"))
        except (TypeError, ValueError):
            return False
        return reported_mode == expected_mode

    def _assert_credential_owner(self, expected_target: str | None) -> str:
        """Fail closed unless the exact bundled Compose service owns 8766."""

        active_target, _ = self._credential_owner_details(expected_target)
        return active_target

    def _credential_owner_details(
        self,
        expected_target: str | None,
    ) -> tuple[str, str]:
        """Return the verified target and immutable container identity."""

        if self._availability_reason is not None:
            self._owner_error()
        try:
            parsed = urlsplit(str(self.upstream.service_url))
            service_port = parsed.port
        except (AttributeError, TypeError, ValueError):
            self._owner_error()
        if (
            parsed.scheme != "http"
            or parsed.hostname != _GUARDRAIL_HOST
            or service_port != int(_GUARDRAIL_HOST_PORT)
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            self._owner_error()

        expected = None if expected_target is None else self._target(expected_target)
        running: dict[str, tuple[ComposeTarget, str]] = {}
        for candidate in _TARGETS.values():
            result = self._ownership_command(
                (
                    *self._compose_prefix(candidate),
                    "ps",
                    "--status",
                    "running",
                    "--quiet",
                    candidate.service_name,
                )
            )
            container_ids = self._container_ids(result.stdout)
            if len(container_ids) > 1:
                self._owner_error()
            if container_ids:
                running[candidate.target_profile] = (candidate, container_ids[0])

        if len(running) != 1:
            self._owner_error()
        active_target, (active, container_id) = next(iter(running.items()))
        if expected is not None and active_target != expected.target_profile:
            self._owner_error()

        active_record = self._inspect_container(container_id)
        self._validate_compose_owner(active, container_id, active_record)

        publisher_result = self._ownership_command(
            (
                self._docker_executable or "docker",
                "ps",
                "--quiet",
                "--filter",
                "status=running",
                "--filter",
                f"publish={_GUARDRAIL_HOST_PORT}",
            )
        )
        publisher_candidates = self._container_ids(publisher_result.stdout)
        publishers: list[str] = []
        for candidate_id in publisher_candidates:
            record = (
                active_record
                if _same_container_id(candidate_id, container_id)
                else self._inspect_container(candidate_id)
            )
            if _publishes_host_port(record, _GUARDRAIL_HOST_PORT):
                publishers.append(candidate_id)
        if (
            len(publishers) != 1
            or not _same_container_id(publishers[0], container_id)
        ):
            self._owner_error()
        return active_target, container_id

    def _compose_prefix(self, target: ComposeTarget) -> tuple[str, ...]:
        return (
            self._docker_executable or "docker",
            "compose",
            "--project-directory",
            str(self.project_root),
            "--file",
            str(self.compose_file),
            "--profile",
            target.compose_profile,
        )

    def _ownership_command(self, argv: Sequence[str]) -> CommandResult:
        try:
            result = self._command_runner(
                argv,
                self.project_root,
                min(self.command_timeout_seconds, 10.0),
            )
        except (OSError, subprocess.SubprocessError):
            self._owner_error()
        if int(result.returncode) != 0:
            self._owner_error()
        return result

    def _container_ids(self, output: object) -> tuple[str, ...]:
        identifiers: list[str] = []
        for line in str(output).splitlines():
            identifier = line.strip().lower()
            if not identifier:
                continue
            if _CONTAINER_ID_PATTERN.fullmatch(identifier) is None:
                self._owner_error()
            if identifier not in identifiers:
                identifiers.append(identifier)
        return tuple(identifiers)

    def _inspect_container(self, container_id: str) -> dict[str, Any]:
        result = self._ownership_command(
            (
                self._docker_executable or "docker",
                "inspect",
                "--type",
                "container",
                container_id,
            )
        )
        try:
            decoded = strict_json_loads(result.stdout)
        except (TypeError, ValueError):
            self._owner_error()
        if (
            not isinstance(decoded, list)
            or len(decoded) != 1
            or not isinstance(decoded[0], dict)
        ):
            self._owner_error()
        return decoded[0]

    def _validate_compose_owner(
        self,
        target: ComposeTarget,
        container_id: str,
        record: dict[str, Any],
    ) -> None:
        self._validate_compose_identity(
            target,
            container_id,
            record,
            expected_running=True,
        )
        if not _has_exact_loopback_binding(record):
            self._owner_error()

    def _validate_compose_identity(
        self,
        target: ComposeTarget,
        container_id: str,
        record: dict[str, Any],
        *,
        expected_running: bool,
    ) -> None:
        inspected_id = str(record.get("Id", "")).strip().lower()
        state = record.get("State")
        config = record.get("Config")
        labels = config.get("Labels") if isinstance(config, dict) else None
        if (
            _CONTAINER_ID_PATTERN.fullmatch(inspected_id) is None
            or not _same_container_id(inspected_id, container_id)
            or not isinstance(state, dict)
            or state.get("Running") is not expected_running
            or not isinstance(labels, dict)
            or labels.get("com.docker.compose.service") != target.service_name
            or labels.get("com.docker.compose.oneoff") not in {"False", "false"}
            or not _same_resolved_path(
                labels.get("com.docker.compose.project.working_dir"),
                self.project_root,
            )
            or not _has_exact_host_config_binding(record)
        ):
            self._owner_error()

    @staticmethod
    def _owner_error() -> None:
        raise GUIControlError(
            (
                "The local AEGIS bearer was not sent because ownership of "
                "127.0.0.1:8766 by the expected Compose service could not be proven."
            ),
            code="credential_owner_unverified",
        )

    def _start_existing(self, target: ComposeTarget, container_id: str) -> None:
        """Restart only the exact stopped container retained for rollback."""

        retained = self._ownership_command(
            (
                *self._compose_prefix(target),
                "ps",
                "--all",
                "--quiet",
                target.service_name,
            )
        )
        retained_ids = self._container_ids(retained.stdout)
        if (
            len(retained_ids) != 1
            or not _same_container_id(retained_ids[0], container_id)
        ):
            self._owner_error()
        record = self._inspect_container(container_id)
        self._validate_compose_identity(
            target,
            container_id,
            record,
            expected_running=False,
        )
        try:
            result = self._command_runner(
                (
                    self._docker_executable or "docker",
                    "start",
                    container_id,
                ),
                self.project_root,
                self.command_timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GUIControlError(
                "The retained previous AEGIS container could not be restarted.",
                code="rollback_container_unavailable",
            ) from exc
        if int(result.returncode) != 0:
            raise GUIControlError(
                "The retained previous AEGIS container could not be restarted.",
                code="rollback_container_unavailable",
            )

    def _compose(
        self,
        target: ComposeTarget,
        action: str,
    ) -> None:
        prefix = self._compose_prefix(target)
        if action == "stop":
            argv = (*prefix, "stop", target.service_name)
        elif action == "start":
            argv = (
                *prefix,
                "up",
                "-d",
                "--no-deps",
                "--build",
                target.service_name,
            )
        else:  # Internal callers use a closed action set.
            raise ValueError(f"Unsupported Compose action: {action!r}.")
        try:
            result = self._command_runner(
                argv,
                self.project_root,
                self.command_timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GUIControlError(
                f"Docker Compose could not {action} {target.service_name}.",
                code="compose_command_failed",
            ) from exc
        if int(result.returncode) != 0:
            raise GUIControlError(
                (
                    f"Docker Compose could not {action} {target.service_name} "
                    f"(exit status {int(result.returncode)})."
                ),
                code="compose_command_failed",
            )

    def _wait_until_ready(self, target_profile: str) -> dict[str, Any]:
        deadline = self._monotonic() + self.startup_timeout_seconds
        last_error: Exception | None = None
        while True:
            try:
                readiness = self.upstream.readiness()
                if (
                    bool(readiness.get("ok"))
                    and readiness.get("target_profile") == target_profile
                ):
                    return readiness
            except Exception as exc:
                last_error = exc
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                message = (
                    f"Target {target_profile!r} did not report ready within "
                    f"{self.startup_timeout_seconds:g} seconds."
                )
                raise GUIControlError(
                    message,
                    code="target_startup_timeout",
                ) from last_error
            self._sleep(min(self.poll_interval_seconds, remaining))

    def _validate_environment(self) -> str | None:
        if not self.project_root.is_dir():
            return "The configured AEGIS project root is not a directory."
        if not self.compose_file.is_file():
            return "The configured AEGIS project root does not contain compose.yaml."
        try:
            compose_source = self.compose_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return "The configured compose.yaml could not be read."
        missing_services = [
            target.service_name
            for target in _TARGETS.values()
            if re.search(
                rf"(?m)^[ \t]+{re.escape(target.service_name)}:[ \t]*$",
                compose_source,
            )
            is None
        ]
        if missing_services:
            return (
                "The configured compose.yaml is not the bundled AEGIS deployment "
                f"(missing services: {missing_services})."
            )
        try:
            service_url = str(self.upstream.service_url)
        except (AttributeError, TypeError, ValueError):
            return "The configured AEGIS service URL is invalid."
        parsed = urlsplit(service_url)
        hostname = parsed.hostname
        if (
            parsed.scheme not in {"http", "https"}
            or hostname is None
            or not _is_loopback_host(hostname)
        ):
            return "Target switching is available only for a local loopback AEGIS service."
        if self._using_default_runner and self._docker_executable is None:
            return "Docker could not be located on PATH or in Docker Desktop."
        return None

    @staticmethod
    def _target(target_profile: str) -> ComposeTarget:
        try:
            return _TARGETS[str(target_profile)]
        except KeyError as exc:
            raise GUIControlError(
                (
                    f"Unknown target profile {target_profile!r}; choose one of "
                    f"{sorted(_TARGETS)}."
                ),
                code="invalid_target",
            ) from exc

    def _set_operation(
        self,
        *,
        operation_id: str,
        stage: str,
        target_profile: str,
        previous_target: str | None,
    ) -> None:
        with self._state_lock:
            self._operation = {
                "id": operation_id,
                "stage": stage,
                "target_profile": target_profile,
                "previous_target": previous_target,
            }

    def _set_operation_stage(self, stage: str) -> None:
        with self._state_lock:
            if self._operation is not None:
                self._operation["stage"] = stage

    def _clear_operation(self) -> None:
        with self._state_lock:
            self._operation = None


def _run_command(
    argv: Sequence[str],
    cwd: Path,
    timeout_seconds: float,
) -> CommandResult:
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd),
        timeout=timeout_seconds,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    return CommandResult(
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _require_positive_finite(value: object, name: str) -> None:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be positive and finite.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be positive and finite.")


def _same_container_id(left: str, right: str) -> bool:
    normalized_left = str(left).strip().lower()
    normalized_right = str(right).strip().lower()
    return normalized_left.startswith(normalized_right) or normalized_right.startswith(
        normalized_left
    )


def _same_resolved_path(raw: object, expected: Path) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return False
    try:
        actual = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return os.path.normcase(str(actual)) == os.path.normcase(str(expected.resolve()))


def _port_binding_entries(record: dict[str, Any], section: str) -> object:
    container_section = record.get(section)
    if not isinstance(container_section, dict):
        return None
    key = "PortBindings" if section == "HostConfig" else "Ports"
    bindings = container_section.get(key)
    if not isinstance(bindings, dict):
        return None
    return bindings.get(_GUARDRAIL_CONTAINER_PORT)


def _has_exact_loopback_binding(record: dict[str, Any]) -> bool:
    return (
        _has_exact_host_config_binding(record)
        and _port_binding_entries(record, "NetworkSettings")
        == [{"HostIp": _GUARDRAIL_HOST, "HostPort": _GUARDRAIL_HOST_PORT}]
        and _count_host_port_bindings(
            record,
            "NetworkSettings",
            _GUARDRAIL_HOST_PORT,
        )
        == 1
    )


def _has_exact_host_config_binding(record: dict[str, Any]) -> bool:
    expected = [{"HostIp": _GUARDRAIL_HOST, "HostPort": _GUARDRAIL_HOST_PORT}]
    return (
        _port_binding_entries(record, "HostConfig") == expected
        and _count_host_port_bindings(record, "HostConfig", _GUARDRAIL_HOST_PORT)
        == 1
    )


def _count_host_port_bindings(
    record: dict[str, Any],
    section: str,
    host_port: str,
) -> int:
    container_section = record.get(section)
    if not isinstance(container_section, dict):
        return 0
    key = "PortBindings" if section == "HostConfig" else "Ports"
    bindings = container_section.get(key)
    if not isinstance(bindings, dict):
        return 0
    return sum(
        1
        for entries in bindings.values()
        if isinstance(entries, list)
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("HostPort")) == host_port
    )


def _publishes_host_port(record: dict[str, Any], host_port: str) -> bool:
    for section in ("HostConfig", "NetworkSettings"):
        container_section = record.get(section)
        if not isinstance(container_section, dict):
            continue
        key = "PortBindings" if section == "HostConfig" else "Ports"
        bindings = container_section.get(key)
        if not isinstance(bindings, dict):
            continue
        for entries in bindings.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and str(entry.get("HostPort")) == host_port:
                    return True
    return False


def _resolve_docker_executable() -> str | None:
    executable = shutil.which("docker")
    if executable is not None:
        return executable
    if os.name != "nt":
        return None
    candidates: list[Path] = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(
            Path(local_app_data)
            / "Programs"
            / "DockerDesktop"
            / "resources"
            / "bin"
            / "docker.exe"
        )
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(
            Path(program_files)
            / "Docker"
            / "Docker"
            / "resources"
            / "bin"
            / "docker.exe"
        )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return None


def _is_loopback_host(hostname: str) -> bool:
    normalized = str(hostname).strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _reported_mode(
    mode_result: dict[str, Any],
    readiness: dict[str, Any],
    fallback: str,
) -> str:
    for candidate in (
        mode_result.get("traffic_mode"),
        readiness.get("traffic_mode"),
        fallback,
    ):
        try:
            return validate_traffic_mode(candidate)
        except (TypeError, ValueError):
            continue
    return fallback


__all__ = [
    "CommandResult",
    "ComposeTarget",
    "ComposeTargetController",
    "GUIControlBusyError",
    "GUIControlError",
    "TargetControlUpstream",
]
