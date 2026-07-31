from __future__ import annotations

import argparse
import json
import signal
from typing import Any

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

    return parser


def _command_demo(args: argparse.Namespace) -> None:
    from AEGIS.demo import evaluate_demo

    _print_json(evaluate_demo(args.text, request_id=args.request_id))


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

    server = create_guardrail_http_server(config.server.host, config.server.port, service)
    readiness = service.readiness_report()
    _print_json(
        {
            "event": "aegis_service_started",
            "name": config.name,
            "host": config.server.host,
            "port": server.server_address[1],
            "authentication_required": service.api_token is not None,
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
        close = getattr(service, "close", None)
        if callable(close):
            close()


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
