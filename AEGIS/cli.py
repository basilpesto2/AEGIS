from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import tempfile
from typing import Any
from urllib.parse import urlencode

from AEGIS import __version__


DEFAULT_CONFIG = "configs/aegis.deployment.container.json"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="Run the AEGIS malicious-prompt guardrail.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="Run a dependency-light installation check.")
    demo.add_argument(
        "text",
        nargs="?",
        default="Explain how rainbows form.",
        help="Text to evaluate with the deterministic demo provider.",
    )
    demo.add_argument("--request-id", default="demo-request")
    demo.set_defaults(func=_command_demo)

    targets = commands.add_parser("targets", help="List supported deployment targets.")
    targets.set_defaults(func=_command_targets)

    prepare = commands.add_parser("prepare", help="Download a target's pinned model files.")
    prepare.add_argument("--target", choices=("llava05b", "qwen25vl3b"), required=True)
    prepare.add_argument("--root", default=".", help="AEGIS project root.")
    prepare.add_argument("--download", action="store_true", help="Download missing files.")
    prepare.add_argument("--token-env", default="HF_TOKEN")
    prepare.set_defaults(func=_command_prepare)

    doctor = commands.add_parser("doctor", help="Check a deployment before starting it.")
    doctor.add_argument("--config", default=DEFAULT_CONFIG)
    doctor.set_defaults(func=_command_doctor)

    serve = commands.add_parser("serve", help="Start the AEGIS HTTP service.")
    serve.add_argument("--config", default=DEFAULT_CONFIG)
    serve.set_defaults(func=_command_serve)

    gui = commands.add_parser(
        "gui",
        help="Start the optional local web dashboard for a running AEGIS service.",
    )
    gui.add_argument(
        "--service-url",
        default="http://127.0.0.1:8766",
        help="HTTP(S) origin of the running AEGIS service.",
    )
    gui.add_argument(
        "--host",
        default="127.0.0.1",
        help="Loopback address for the local dashboard.",
    )
    gui.add_argument("--port", type=int, default=8767, help="Dashboard port.")
    gui.add_argument(
        "--history-db",
        default=str(Path.home() / ".aegis" / "gui-history.sqlite3"),
        help=(
            "SQLite file for raw GUI prompt, image, result, and error history. "
            "Created with restricted permissions where supported."
        ),
    )
    gui.add_argument(
        "--state-file",
        default=None,
        help=(
            "Private GUI shutdown-state JSON file (default: gui-state.json next "
            "to --history-db)."
        ),
    )
    gui.add_argument(
        "--history-max-records",
        type=int,
        default=1000,
        help="Maximum retained GUI history records.",
    )
    gui.add_argument(
        "--history-max-age-days",
        type=float,
        default=30.0,
        help="Maximum age of retained GUI history records in days.",
    )
    gui.add_argument(
        "--history-max-bytes",
        type=int,
        default=268435456,
        help="Maximum retained GUI history payload size in bytes.",
    )
    gui.add_argument(
        "--api-token-env",
        default="AEGIS_API_TOKEN",
        help="Environment variable containing the upstream AEGIS bearer token.",
    )
    gui.add_argument(
        "--admin-token-env",
        default="AEGIS_ADMIN_TOKEN",
        help=(
            "Environment variable containing the distinct bearer token for upstream "
            "runtime administration. Runtime controls are disabled when it is unset."
        ),
    )
    gui.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Maximum time to wait for one upstream evaluation.",
    )
    gui.add_argument(
        "--project-root",
        default=".",
        help=(
            "AEGIS checkout containing compose.yaml for local target switching "
            "(default: current directory)."
        ),
    )
    gui.add_argument(
        "--target-switch-timeout-seconds",
        type=float,
        default=1200.0,
        help="Maximum time to wait for a Compose target switch.",
    )
    gui.add_argument(
        "--max-http-connections",
        type=int,
        default=32,
        help="Maximum simultaneous dashboard HTTP connections.",
    )
    gui.add_argument(
        "--request-read-timeout-seconds",
        type=float,
        default=15.0,
        help="Absolute deadline for reading one dashboard HTTP request.",
    )
    gui.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the dashboard in the default browser after startup.",
    )
    gui.set_defaults(func=_command_gui)

    return parser


def _command_demo(args: argparse.Namespace) -> None:
    from AEGIS.demo import evaluate_demo

    report = evaluate_demo(args.text, request_id=args.request_id)
    _print_json(report)
    decision = report.get("decision")
    if not isinstance(decision, dict) or decision.get("verdict") not in {
        "benign",
        "malicious",
    }:
        raise SystemExit(1)


def _command_targets(args: argparse.Namespace) -> None:
    from AEGIS.target_profiles import list_target_profiles

    _print_json(list_target_profiles())


def _command_prepare(args: argparse.Namespace) -> None:
    from AEGIS.model_prepare import prepare_target_model

    report = prepare_target_model(
        args.target,
        root=args.root,
        download=args.download,
        token_env=args.token_env,
    )
    _print_json(report)
    if args.download and not bool(report["ready"]):
        raise SystemExit(1)


def _command_doctor(args: argparse.Namespace) -> None:
    from AEGIS.deployment_config import deployment_doctor, load_deployment_config

    report = deployment_doctor(load_deployment_config(args.config))
    _print_json(report)
    if not bool(report["ok"]):
        raise SystemExit(1)


def _command_serve(args: argparse.Namespace) -> None:
    from AEGIS.deployment_config import build_service_from_config, load_deployment_config
    from AEGIS.http_server import create_guardrail_http_server

    config = load_deployment_config(args.config)
    service, startup = build_service_from_config(config)
    if not service.ready:
        raise RuntimeError(
            "Deployment readiness failed: "
            + json.dumps(service.readiness_report(), sort_keys=True)
        )

    server = create_guardrail_http_server(
        config.server.host,
        config.server.port,
        service,
        max_http_connections=config.server.max_http_connections,
        request_read_timeout_seconds=config.server.request_read_timeout_seconds,
    )
    readiness = service.readiness_report()
    _print_json(
        {
            "event": "aegis_service_started",
            "name": config.name,
            "host": config.server.host,
            "port": server.server_address[1],
            "authentication_required": service.api_token is not None,
            "admin_authentication_configured": service.admin_token is not None,
            "warmup_completed": startup["warmup"] is not None,
            "traffic_mode": config.traffic_mode,
            "evidence_session_id": readiness.get("evidence_session_id"),
            "audit_path": readiness.get("audit_path"),
        }
    )

    handled_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        handled_signals.append(signal.SIGBREAK)
    previous_handlers = {
        handled_signal: signal.getsignal(handled_signal)
        for handled_signal in handled_signals
    }

    def stop_on_signal(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    for handled_signal in handled_signals:
        signal.signal(handled_signal, stop_on_signal)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)
        server.server_close()
        server.wait_for_request_threads(
            max(float(config.server.inference_timeout_seconds) + 15.0, 30.0)
        )
        close = getattr(service, "close", None)
        if callable(close):
            close()


def _command_gui(args: argparse.Namespace) -> None:
    from AEGIS.gui_server import GUIServerConfig, create_gui_http_server

    api_token = os.getenv(args.api_token_env) or None
    admin_token = os.getenv(args.admin_token_env) or None
    history_path = Path(args.history_db).expanduser().resolve()
    state_path = (
        Path(args.state_file).expanduser().resolve()
        if args.state_file
        else history_path.parent / "gui-state.json"
    )
    if state_path == history_path:
        raise ValueError("The GUI state file must be distinct from the history database.")
    config = GUIServerConfig(
        service_url=args.service_url,
        history_path=history_path,
        history_max_records=args.history_max_records,
        history_max_age_days=args.history_max_age_days,
        history_max_bytes=args.history_max_bytes,
        api_token=api_token,
        admin_token=admin_token,
        upstream_timeout_seconds=args.timeout_seconds,
        project_root=Path(args.project_root).expanduser().resolve(),
        target_switch_timeout_seconds=args.target_switch_timeout_seconds,
        max_http_connections=args.max_http_connections,
        request_read_timeout_seconds=args.request_read_timeout_seconds,
    )
    server = create_gui_http_server(args.host, args.port, config)
    bound_host, bound_port = server.server_address[:2]
    browser_host = (
        "127.0.0.1"
        if str(bound_host) in {"0.0.0.0", "::"}
        else str(bound_host)
    )
    if ":" in browser_host and not browser_host.startswith("["):
        browser_host = f"[{browser_host}]"
    dashboard_url = f"http://{browser_host}:{bound_port}/"
    bootstrap_url = f"{dashboard_url}#{urlencode({'session': config.session_token})}"
    state_published = False
    previous_handlers: dict[int, Any] = {}
    try:
        _publish_gui_state(
            state_path,
            {
                "schema_version": 1,
                "pid": os.getpid(),
                "host": str(bound_host),
                "port": int(bound_port),
                "shutdown_token": config.shutdown_token,
            },
        )
        state_published = True
        _print_json(
            {
                "event": "aegis_gui_started",
                "url": bootstrap_url,
                "dashboard_origin": dashboard_url,
                "service_url": config.service_url,
                "authentication_configured": api_token is not None,
                "admin_authentication_configured": admin_token is not None,
                "history_path": str(config.history_path),
                "state_path": str(state_path),
                "history_contains_raw_inputs": True,
                "history_retention": {
                    "max_records": config.history_max_records,
                    "max_age_days": config.history_max_age_days,
                    "max_bytes": config.history_max_bytes,
                },
                "project_root": str(config.project_root),
            }
        )
        if args.open_browser:
            import webbrowser

            webbrowser.open_new_tab(bootstrap_url)

        handled_signals = [signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            handled_signals.append(signal.SIGBREAK)
        previous_handlers = {
            handled_signal: signal.getsignal(handled_signal)
            for handled_signal in handled_signals
        }

        def stop_on_signal(signum: int, frame: Any) -> None:
            raise KeyboardInterrupt

        for handled_signal in handled_signals:
            signal.signal(handled_signal, stop_on_signal)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            for handled_signal, previous_handler in previous_handlers.items():
                signal.signal(handled_signal, previous_handler)
        finally:
            try:
                server.begin_drain_and_wait()
            finally:
                try:
                    if state_published:
                        _remove_gui_state(
                            state_path,
                            pid=os.getpid(),
                            shutdown_token=config.shutdown_token,
                        )
                finally:
                    server.server_close()


def _publish_gui_state(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish one process-owned, private GUI shutdown credential."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        try:
            os.chmod(temporary_path, 0o600)
        except OSError:
            pass
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _remove_gui_state(path: Path, *, pid: int, shutdown_token: str) -> None:
    """Remove only the state record still owned by this GUI process."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    if payload.get("pid") != int(pid) or payload.get("shutdown_token") != shutdown_token:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
