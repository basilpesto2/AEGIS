from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import re
import shutil
import subprocess
from threading import Lock
import time
from typing import Any, Callable, Protocol, Sequence
from urllib.parse import urlsplit
import uuid

from AEGIS.audit import validate_traffic_mode
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

    def set_traffic_mode(self, traffic_mode: str) -> dict[str, Any]: ...


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
        startup_timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 1.0,
        command_timeout_seconds: float = 120.0,
        command_runner: CommandRunner | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive.")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive.")
        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive.")

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
        self._operation_lock = Lock()
        self._state_lock = Lock()
        self._operation: dict[str, object] | None = None
        self._availability_reason = self._validate_environment()

    @property
    def available(self) -> bool:
        return self._availability_reason is None

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return self._operation is not None

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
        if not self._operation_lock.acquire(blocking=False):
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
            previous_mode = self._active_mode(previous)
            if same_target:
                self._set_operation_stage("setting_traffic_mode")
                mode_result = self.upstream.set_traffic_mode(mode)
                readiness = self._wait_until_ready(target.target_profile)
                return {
                    "operation_id": operation_id,
                    "changed_target": False,
                    "active_target": target.target_profile,
                    "traffic_mode": _reported_mode(mode_result, readiness, mode),
                    "rollback_succeeded": None,
                }

            if previous is not None:
                self._set_operation_stage("stopping_previous")
                self._compose(previous, "stop")
            else:
                self._set_operation_stage("stopping_conflicts")
                for candidate in _TARGETS.values():
                    if candidate != target:
                        self._compose(candidate, "stop")

            try:
                self._set_operation_stage("starting_target")
                self._compose(target, "start")
                self._set_operation_stage("waiting_for_readiness")
                readiness = self._wait_until_ready(target.target_profile)
                self._set_operation_stage("setting_traffic_mode")
                mode_result = self.upstream.set_traffic_mode(mode)
                readiness = self._wait_until_ready(target.target_profile)
            except Exception as exc:
                rollback_succeeded = self._rollback(
                    target,
                    previous,
                    previous_mode,
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
    ) -> bool:
        self._set_operation_stage("rolling_back")
        try:
            self._compose(attempted, "stop")
            if previous is None:
                return True
            self._compose(previous, "start")
            self._wait_until_ready(previous.target_profile)
            self.upstream.set_traffic_mode(previous_mode)
            self._wait_until_ready(previous.target_profile)
            return True
        except Exception:
            return False

    def _active_mode(self, previous: ComposeTarget | None) -> str:
        if previous is None:
            return "shadow"
        try:
            readiness = self.upstream.readiness()
            if readiness.get("target_profile") == previous.target_profile:
                return validate_traffic_mode(readiness.get("traffic_mode", "shadow"))
        except Exception:
            pass
        return "shadow"

    def _compose(
        self,
        target: ComposeTarget,
        action: str,
    ) -> None:
        prefix = (
            "docker",
            "compose",
            "--project-directory",
            str(self.project_root),
            "--file",
            str(self.compose_file),
            "--profile",
            target.compose_profile,
        )
        if action == "stop":
            argv = (*prefix, "stop", target.service_name)
        elif action == "start":
            argv = (*prefix, "up", "-d", "--no-deps", target.service_name)
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
        if self._using_default_runner and shutil.which("docker") is None:
            return "Docker is not available on PATH."
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
