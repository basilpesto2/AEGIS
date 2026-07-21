from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid

from AEGIS.audit import (
    AuditLogConfig,
    PrivacySafeAuditLogger,
    validate_traffic_mode,
)
from AEGIS.detector_artifact import load_detector_artifact
from AEGIS.guardrail import GuardrailPolicy
from AEGIS.http_server import (
    GuardrailHTTPService,
    provider_readiness_report,
    require_token_for_bind,
)
from AEGIS.provider_contract import load_provider
from AEGIS.readiness import sha256_file
from AEGIS.service import RequestLimits
from AEGIS.service import evaluate_request_payload
from AEGIS.system_resources import (
    ResourceRequirements,
    evaluate_resource_requirements,
    inspect_model_cache,
    inspect_system_resources,
    resolve_model_storage_path,
)


DEPLOYMENT_CONFIG_SCHEMA_VERSION = 1
DEPLOYABLE_ERROR_ACTIONS = frozenset({"review", "block"})


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8766
    api_token_env: str = "AEGIS_API_TOKEN"
    max_body_bytes: int = 12 * 1024 * 1024
    max_concurrent_requests: int = 1
    worker_mode: str = "in_process"
    inference_timeout_seconds: float = 60.0
    worker_startup_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("server.host cannot be empty.")
        if not 0 <= self.port <= 65_535:
            raise ValueError("server.port must be in [0, 65535].")
        if not self.api_token_env.strip():
            raise ValueError("server.api_token_env cannot be empty.")
        if self.max_body_bytes <= 0 or self.max_concurrent_requests <= 0:
            raise ValueError("server size and concurrency limits must be positive.")
        if self.worker_mode not in {"in_process", "process"}:
            raise ValueError("server.worker_mode must be 'in_process' or 'process'.")
        if self.worker_mode == "process" and self.max_concurrent_requests != 1:
            raise ValueError(
                "server.worker_mode='process' currently requires max_concurrent_requests=1."
            )
        if self.inference_timeout_seconds <= 0:
            raise ValueError("server.inference_timeout_seconds must be positive.")
        if self.worker_startup_timeout_seconds <= 0:
            raise ValueError("server.worker_startup_timeout_seconds must be positive.")


@dataclass(frozen=True)
class DeploymentConfig:
    name: str
    detector_path: Path
    provider_spec: str
    provider_options: dict[str, object]
    policy: GuardrailPolicy
    server: ServerConfig
    request_limits: RequestLimits
    config_path: Path
    cache_dir: Path | None = None
    warmup_request_path: Path | None = None
    require_model_cache: bool = False
    require_cuda: bool = False
    target_profile: str | None = None
    traffic_mode: str = "shadow"
    evidence_session_mode: str = "fixed"
    evidence_session_id: str | None = None
    audit: AuditLogConfig = AuditLogConfig()
    base_dir: Path = Path(".")
    resources: ResourceRequirements = ResourceRequirements()


def load_deployment_config(path: str | Path) -> DeploymentConfig:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Deployment config must contain a JSON object.")
    _require_keys(
        payload,
        required={"schema_version", "name", "detector", "provider", "policy", "server"},
        optional={
            "base_dir",
            "request_limits",
            "cache_dir",
            "warmup_request_json",
            "require_model_cache",
            "require_cuda",
            "provider_options",
            "target_profile",
            "resources",
            "traffic_mode",
            "evidence_session_mode",
            "evidence_session_id",
            "audit",
        },
        context="deployment config",
    )
    if payload["schema_version"] != DEPLOYMENT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported deployment config schema {payload['schema_version']!r}; "
            f"expected {DEPLOYMENT_CONFIG_SCHEMA_VERSION}."
        )
    name = str(payload["name"]).strip()
    provider_spec = str(payload["provider"]).strip()
    if not name or not provider_spec:
        raise ValueError("Deployment name and provider cannot be empty.")
    base_dir = (config_path.parent / str(payload.get("base_dir", "."))).resolve()
    policy_payload = _mapping(payload["policy"], "policy")
    server_payload = _mapping(payload["server"], "server")
    limits_payload = _mapping(payload.get("request_limits", {}), "request_limits")
    provider_options = _mapping(payload.get("provider_options", {}), "provider_options")
    resources_payload = _mapping(payload.get("resources", {}), "resources")
    audit_payload = _mapping(payload.get("audit", {}), "audit")
    _require_keys(
        policy_payload,
        required=set(),
        optional={
            "block_threshold",
            "review_threshold",
            "review_margin",
            "action_on_error",
            "require_matching_provenance",
            "hash_images",
            "fingerprint_key_env",
        },
        context="policy",
    )
    _require_keys(
        server_payload,
        required=set(),
        optional={
            "host",
            "port",
            "api_token_env",
            "max_body_bytes",
            "max_concurrent_requests",
            "worker_mode",
            "inference_timeout_seconds",
            "worker_startup_timeout_seconds",
        },
        context="server",
    )
    _require_keys(
        limits_payload,
        required=set(),
        optional={field.name for field in RequestLimits.__dataclass_fields__.values()},
        context="request_limits",
    )
    _require_keys(
        resources_payload,
        required=set(),
        optional={field.name for field in ResourceRequirements.__dataclass_fields__.values()},
        context="resources",
    )
    _require_keys(
        audit_payload,
        required=set(),
        optional={"path", "max_bytes", "backup_count", "fsync"},
        context="audit",
    )
    policy = GuardrailPolicy(**policy_payload)
    server = ServerConfig(**server_payload)
    request_limits = RequestLimits(**limits_payload)
    cache_value = payload.get("cache_dir")
    warmup_value = payload.get("warmup_request_json")
    audit_path_value = audit_payload.pop("path", None)
    evidence_session_mode = _validate_evidence_session_mode(
        payload.get("evidence_session_mode", "fixed")
    )
    evidence_session_id = _validate_evidence_session_id(payload.get("evidence_session_id"))
    if evidence_session_mode == "runtime" and evidence_session_id is not None:
        raise ValueError(
            "Runtime evidence sessions must not configure a fixed evidence_session_id."
        )
    return DeploymentConfig(
        name=name,
        detector_path=_resolve_path(base_dir, payload["detector"]),
        provider_spec=provider_spec,
        provider_options=provider_options,
        policy=policy,
        server=server,
        request_limits=request_limits,
        config_path=config_path,
        cache_dir=None if cache_value is None else _resolve_path(base_dir, cache_value),
        warmup_request_path=(
            None if warmup_value is None else _resolve_path(base_dir, warmup_value)
        ),
        require_model_cache=bool(payload.get("require_model_cache", False)),
        require_cuda=bool(payload.get("require_cuda", False)),
        target_profile=(
            None if payload.get("target_profile") is None else str(payload["target_profile"])
        ),
        traffic_mode=validate_traffic_mode(payload.get("traffic_mode", "shadow")),
        evidence_session_mode=evidence_session_mode,
        evidence_session_id=evidence_session_id,
        audit=AuditLogConfig(
            path=(None if audit_path_value is None else _resolve_path(base_dir, audit_path_value)),
            **audit_payload,
        ),
        base_dir=base_dir,
        resources=ResourceRequirements(**resources_payload),
    )


def build_service_from_config(
    config: DeploymentConfig,
) -> tuple[Any, dict[str, object]]:
    if config.policy.action_on_error not in DEPLOYABLE_ERROR_ACTIONS:
        raise RuntimeError(
            "Deployment refused because policy.action_on_error would fail open; "
            "use 'review' (recommended) or 'block'."
        )
    preflight = deployment_doctor(config)
    if not bool(preflight["ok"]):
        failed = [
            {
                "name": item["name"],
                "detail": item["detail"],
            }
            for item in preflight["checks"]
            if item["required"] and not item["ok"]
        ]
        raise RuntimeError(
            "Deployment preflight failed before model loading: "
            + json.dumps(failed, sort_keys=True)
        )
    api_token = os.getenv(config.server.api_token_env) or None
    require_token_for_bind(config.server.host, api_token)
    warmup_payload = None
    if config.warmup_request_path is not None:
        warmup_payload = json.loads(config.warmup_request_path.read_text(encoding="utf-8"))
        if not isinstance(warmup_payload, dict):
            raise ValueError("Warmup request JSON must contain an object.")
    if config.server.worker_mode == "process":
        from AEGIS.isolated_service import ProcessIsolatedGuardrailService

        isolated_service = ProcessIsolatedGuardrailService(
            artifact_path=config.detector_path,
            provider_spec=config.provider_spec,
            provider_options=_provider_options(config),
            policy=config.policy,
            limits=config.request_limits,
            api_token=api_token,
            max_body_bytes=config.server.max_body_bytes,
            max_concurrent_requests=config.server.max_concurrent_requests,
            inference_timeout_seconds=config.server.inference_timeout_seconds,
            worker_startup_timeout_seconds=config.server.worker_startup_timeout_seconds,
            warmup_payload=warmup_payload,
            traffic_mode=config.traffic_mode,
            audit_logger=_audit_logger(config),
        )
        return isolated_service, {"warmup": isolated_service.startup_warmup}

    artifact = load_detector_artifact(config.detector_path)
    provider = load_provider(config.provider_spec, _provider_options(config))
    service = GuardrailHTTPService(
        artifact=artifact,
        provider=provider,
        policy=config.policy,
        limits=config.request_limits,
        api_token=api_token,
        max_body_bytes=config.server.max_body_bytes,
        max_concurrent_requests=config.server.max_concurrent_requests,
        traffic_mode=config.traffic_mode,
        audit_logger=_audit_logger(config),
    )
    warmup = None
    if warmup_payload is not None:
        warmup = service.evaluate(warmup_payload)
        if any(
            decision.get("verdict") == "guardrail_error" for decision in warmup.get("decisions", [])
        ):
            raise RuntimeError("Warmup request produced a guardrail_error decision.")
    return service, {"warmup": warmup}


def deployment_doctor(
    config: DeploymentConfig,
    *,
    probe_path: str | Path | None = None,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "required": required, "detail": detail})

    check(
        "policy_action_on_error_fail_closed",
        config.policy.action_on_error in DEPLOYABLE_ERROR_ACTIONS,
        (
            f"action_on_error={config.policy.action_on_error!r}, "
            f"allowed={sorted(DEPLOYABLE_ERROR_ACTIONS)!r}"
        ),
    )

    check(
        "detector_exists",
        config.detector_path.exists(),
        f"path={config.detector_path}",
    )
    artifact = None
    provider = None
    cuda_devices: tuple[dict[str, object], ...] = ()
    if config.detector_path.exists():
        try:
            artifact = load_detector_artifact(config.detector_path)
            check("detector_loads", True, f"feature_dim={artifact.feature_dim}")
        except Exception as exc:
            check("detector_loads", False, str(exc))
    if artifact is not None:
        check(
            "artifact_model_revision_pinned",
            bool(artifact.model_revision),
            f"model_revision={artifact.model_revision!r}",
            required=False,
        )
        check(
            "artifact_tokenizer_revision_pinned",
            bool(artifact.tokenizer_revision),
            f"tokenizer_revision={artifact.tokenizer_revision!r}",
            required=False,
        )
        check(
            "artifact_preprocessing_fingerprinted",
            bool(artifact.preprocessing_sha256),
            f"preprocessing_sha256={artifact.preprocessing_sha256!r}",
            required=False,
        )
    if config.target_profile is not None:
        try:
            from AEGIS.target_profiles import get_target_profile

            target_profile = get_target_profile(config.target_profile)
            check(
                "target_profile_provider_matches",
                config.provider_spec == target_profile.provider,
                f"configured={config.provider_spec!r}, profile={target_profile.provider!r}",
            )
            check(
                "target_profile_environment_overrides_disabled",
                config.provider_options.get("environment_overrides") is False,
                (f"environment_overrides={config.provider_options.get('environment_overrides')!r}"),
            )
            check(
                "target_profile_local_model_loading",
                config.provider_options.get("local_files_only") is True,
                (f"local_files_only={config.provider_options.get('local_files_only')!r}"),
            )
            from AEGIS.target_assets import resolve_target_asset

            detector_asset = resolve_target_asset(
                target_profile.detector,
                source_root=config.base_dir,
            )
            expected_detector = detector_asset.path
            check(
                "target_profile_detector_matches",
                expected_detector is not None and config.detector_path == expected_detector,
                f"configured={config.detector_path}, profile={expected_detector}",
            )
            check(
                "target_profile_cuda_requirement_matches",
                config.require_cuda == target_profile.require_cuda,
                (f"configured={config.require_cuda}, profile={target_profile.require_cuda}"),
            )
            check(
                "target_profile_model_cache_required",
                config.require_model_cache,
                f"require_model_cache={config.require_model_cache}",
            )
            check(
                "target_profile_warmup_configured",
                config.warmup_request_path is not None,
                f"warmup_request_path={config.warmup_request_path}",
            )
            check(
                "target_profile_process_isolation",
                config.server.worker_mode == "process",
                (f"worker_mode={config.server.worker_mode!r}, required='process'"),
            )
            profile_options = {
                name: value
                for name, value in target_profile.provider_options.items()
                if name not in {"model_revision", "tokenizer_revision"}
            }
            option_mismatches = {
                name: {
                    "configured": config.provider_options.get(name),
                    "profile": value,
                }
                for name, value in profile_options.items()
                if config.provider_options.get(name) != value
            }
            check(
                "target_profile_provider_options_match",
                not option_mismatches,
                f"mismatches={option_mismatches!r}",
            )
            pinned_revision_names = [
                name
                for name in ("model_revision", "tokenizer_revision")
                if target_profile.provider_options.get(name)
            ]
            missing_revisions = [
                name for name in pinned_revision_names if not config.provider_options.get(name)
            ]
            check(
                "target_profile_required_revisions_pinned",
                not missing_revisions,
                f"missing={missing_revisions!r}",
            )
            configured_resources = asdict(config.resources)
            resource_shortfalls = {
                name: {
                    "configured": int(configured_resources.get(name, 0)),
                    "profile_minimum": int(required),
                }
                for name, required in target_profile.resources.items()
                if int(configured_resources.get(name, 0)) < int(required)
            }
            check(
                "target_profile_resource_budget_matches",
                not resource_shortfalls,
                f"shortfalls={resource_shortfalls!r}",
            )
            check(
                "target_profile_release_status",
                target_profile.status in {"operational_candidate", "operational"},
                f"status={target_profile.status!r}; caveats={list(target_profile.caveats)!r}",
                required=False,
            )
        except ValueError as exc:
            check("target_profile_valid", False, str(exc))
    try:
        provider = load_provider(config.provider_spec, _provider_options(config))
        check("provider_imports", True, f"provider={config.provider_spec}")
    except Exception as exc:
        check("provider_imports", False, str(exc))
    if artifact is not None and provider is not None:
        readiness = provider_readiness_report(
            artifact,
            provider,
            require_matching_provenance=config.policy.require_matching_provenance,
        )
        check("provider_readiness", bool(readiness["ok"]), json.dumps(readiness))

    dependencies = _dependency_names(None if artifact is None else artifact.model_family)
    for dependency in dependencies:
        available = importlib.util.find_spec(dependency) is not None
        check(
            f"dependency:{dependency}",
            available,
            "installed" if available else "missing",
        )

    if artifact is not None and artifact.model_family in {
        "qwen25_vl",
        "llava_onevision",
    }:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            device_count = int(torch.cuda.device_count()) if cuda_available else 0
            detail = {
                "torch_version": str(torch.__version__),
                "cuda_available": cuda_available,
                "device_count": device_count,
                "devices": [],
            }
            if cuda_available:
                for index in range(device_count):
                    properties = torch.cuda.get_device_properties(index)
                    detail["devices"].append(
                        {
                            "index": index,
                            "name": str(properties.name),
                            "total_memory_bytes": int(properties.total_memory),
                        }
                    )
            cuda_devices = tuple(detail["devices"])
            check(
                "cuda_runtime",
                cuda_available or not config.require_cuda,
                json.dumps(detail, sort_keys=True),
                required=config.require_cuda,
            )
        except Exception as exc:
            check(
                "cuda_runtime",
                not config.require_cuda,
                str(exc),
                required=config.require_cuda,
            )

    cache_required = config.require_model_cache
    if config.cache_dir is not None:
        cache_exists = config.cache_dir.exists() and config.cache_dir.is_dir()
        check(
            "model_cache_dir",
            cache_exists or not cache_required,
            f"path={config.cache_dir}, exists={cache_exists}",
            required=cache_required,
        )
    elif cache_required:
        check("model_cache_dir", False, "cache_dir is required but not configured")

    model_storage = resolve_model_storage_path(
        model_id=(
            None
            if provider is None
            else str(
                getattr(provider, "runtime_model_id", None) or getattr(provider, "model_id", "")
            )
        ),
        cache_dir=config.cache_dir,
        root=config.base_dir,
    )
    model_cache_report = inspect_model_cache(
        model_storage,
        model_family=("" if provider is None else str(getattr(provider, "model_family", ""))),
        revision=(None if provider is None else str(getattr(provider, "model_revision", ""))),
    )
    check(
        "model_cache_integrity",
        bool(model_cache_report["ok"]) or not cache_required,
        json.dumps(model_cache_report, sort_keys=True),
        required=cache_required,
    )
    resource_snapshot = inspect_system_resources(
        disk_path=config.cache_dir or config.config_path.parent,
        model_path=model_storage,
        cuda_probe=lambda: cuda_devices,
    )
    resource_report = evaluate_resource_requirements(config.resources, resource_snapshot)
    resource_required = any(value > 0 for value in asdict(config.resources).values())
    check(
        "resource_preflight",
        bool(resource_report["ok"]),
        json.dumps(resource_report, sort_keys=True),
        required=resource_required,
    )

    if config.warmup_request_path is not None:
        warmup_exists = config.warmup_request_path.is_file()
        check(
            "warmup_request_exists",
            warmup_exists,
            f"path={config.warmup_request_path}",
        )
        if warmup_exists:
            try:
                warmup_payload = json.loads(config.warmup_request_path.read_text(encoding="utf-8"))
                check(
                    "warmup_request_valid_json",
                    isinstance(warmup_payload, dict),
                    "warmup request contains a JSON object",
                )
            except (OSError, json.JSONDecodeError) as exc:
                check("warmup_request_valid_json", False, str(exc))

    token = os.getenv(config.server.api_token_env) or None
    try:
        require_token_for_bind(config.server.host, token)
        check(
            "bind_authentication",
            True,
            f"host={config.server.host}, token_configured={token is not None}",
        )
    except ValueError as exc:
        check("bind_authentication", False, str(exc))

    if config.policy.fingerprint_key_env is not None:
        fingerprint_key = os.getenv(config.policy.fingerprint_key_env)
        check(
            "fingerprint_hmac_key",
            bool(fingerprint_key),
            f"env={config.policy.fingerprint_key_env}, configured={bool(fingerprint_key)}",
        )

    audit_required = config.target_profile is not None
    check(
        "privacy_safe_audit_configured",
        config.audit.path is not None or not audit_required,
        f"path={config.audit.path}, traffic_mode={config.traffic_mode!r}",
        required=audit_required,
    )
    evidence_session_required = audit_required and config.audit.path is not None
    evidence_identity_configured = (
        config.evidence_session_mode == "runtime" or config.evidence_session_id is not None
    )
    check(
        "evidence_session_id_configured",
        evidence_identity_configured or not evidence_session_required,
        (
            f"evidence_session_mode={config.evidence_session_mode!r}, "
            f"evidence_session_id={config.evidence_session_id!r}, "
            f"target_profile={config.target_profile!r}, audit_path={config.audit.path}"
        ),
        required=evidence_session_required,
    )
    runtime_template_valid = config.evidence_session_mode != "runtime" or (
        config.audit.path is not None and "{evidence_session_id}" in str(config.audit.path)
    )
    check(
        "runtime_evidence_session_path_template",
        runtime_template_valid,
        (
            f"mode={config.evidence_session_mode!r}, path={config.audit.path}, "
            "placeholder={evidence_session_id}"
        ),
        required=evidence_session_required,
    )
    if config.audit.path is not None:
        audit_parent = _nearest_existing_parent(config.audit.path.parent)
        try:
            with tempfile.NamedTemporaryFile(dir=audit_parent):
                pass
            check(
                "audit_directory_writable",
                True,
                f"path={config.audit.path.parent}, existing_parent={audit_parent}",
            )
        except OSError as exc:
            check("audit_directory_writable", False, str(exc))

    try:
        with tempfile.NamedTemporaryFile(dir=config.config_path.parent):
            pass
        check("config_directory_writable", True, f"path={config.config_path.parent}")
    except OSError as exc:
        check("config_directory_writable", False, str(exc))

    probe_report = None
    if probe_path is not None and artifact is not None and provider is not None:
        try:
            payload = json.loads(Path(probe_path).read_text(encoding="utf-8"))
            probe_report = evaluate_request_payload(
                artifact,
                provider,
                payload,
                policy=config.policy,
                limits=config.request_limits,
                include_error_details=True,
            )
            probe_ok = not any(
                decision.get("verdict") == "guardrail_error"
                for decision in probe_report.get("decisions", [])
            )
            check("end_to_end_probe", probe_ok, f"decision_count={len(probe_report['decisions'])}")
        except Exception as exc:
            check("end_to_end_probe", False, str(exc))

    failed_required = [str(item["name"]) for item in checks if item["required"] and not item["ok"]]
    remediation = [
        {
            "check": name,
            "action": _deployment_doctor_remediation(name, config),
        }
        for name in failed_required
    ]
    if failed_required:
        next_actions: list[dict[str, object]] = [
            {
                "name": "resolve_blockers",
                "blocked_checks": failed_required,
                "purpose": "Apply each remediation without weakening fail-closed policy.",
            },
            {
                "name": "rerun_doctor",
                "argv": ["aegis", "doctor", "--config", str(config.config_path)],
                "purpose": "Confirm every required check passes before serving.",
            },
        ]
    else:
        next_actions = [
            {
                "name": "serve_shadow",
                "argv": ["aegis", "serve", "--config", str(config.config_path)],
                "purpose": "Start the guardrail in its configured traffic mode.",
            }
        ]
    return {
        "ok": not failed_required,
        "config": str(config.config_path),
        "name": config.name,
        "checks": checks,
        "summary": {
            "required": sum(bool(item["required"]) for item in checks),
            "required_passed": sum(bool(item["required"] and item["ok"]) for item in checks),
            "optional_warnings": sum(
                bool(not item["required"] and not item["ok"]) for item in checks
            ),
        },
        "failed_required_checks": failed_required,
        "remediation": remediation,
        "next_actions": next_actions,
        "warnings": [
            {"name": item["name"], "detail": item["detail"]}
            for item in checks
            if not item["required"] and not item["ok"]
        ],
        "probe": probe_report,
    }


def _deployment_doctor_remediation(
    check_name: str,
    config: DeploymentConfig,
) -> str:
    if check_name == "policy_action_on_error_fail_closed":
        return "Set policy.action_on_error to 'review' or 'block'; never use 'allow'."
    if check_name in {"detector_exists", "detector_loads"}:
        return f"Restore and verify the detector artifact at {config.detector_path}."
    if check_name.startswith("target_profile_"):
        config_name = (
            "configs/aegis.llava.deployment.container.json"
            if config.target_profile == "llava05b"
            else "configs/aegis.deployment.container.json"
        )
        return (
            f"Restore the shipped {config_name} file and preserve its detector, provider, "
            "resource, and revision settings."
        )
    if check_name in {"provider_imports", "provider_readiness"} or check_name.startswith(
        "dependency:"
    ):
        return "Install the complete AEGIS runtime wheel and its pinned target dependencies."
    if check_name == "cuda_runtime":
        return "Install/enable the NVIDIA runtime and verify torch can see the required GPU."
    if check_name in {"model_cache_dir", "model_cache_integrity"}:
        if config.target_profile:
            return f"Run `aegis prepare --target {config.target_profile} --download`, then rerun doctor."
        return "Populate the configured pinned model cache and verify its manifest."
    if check_name.startswith("warmup_request"):
        return "Restore a bounded JSON warmup request compatible with the configured provider."
    if check_name == "bind_authentication":
        return f"Set a strong secret in {config.server.api_token_env} or bind only to loopback."
    if check_name == "fingerprint_hmac_key":
        return f"Set an independent high-entropy secret in {config.policy.fingerprint_key_env}."
    if check_name in {
        "privacy_safe_audit_configured",
        "evidence_session_id_configured",
        "runtime_evidence_session_path_template",
        "audit_directory_writable",
    }:
        return "Restore the target's privacy-safe audit/session settings and writable audit path."
    if check_name == "resource_preflight":
        return "Free or provision the RAM, disk, model storage, and VRAM required by the target."
    if check_name == "config_directory_writable":
        return "Move the config to an operator-owned writable deployment directory."
    if check_name == "end_to_end_probe":
        return "Inspect the bounded probe and provider error, then rerun doctor before serving."
    return "Inspect this check's detail, restore the pinned deployment contract, and rerun doctor."


def _dependency_names(model_family: str | None) -> tuple[str, ...]:
    common = ("numpy", "pandas", "PIL")
    if model_family == "qwen25_vl":
        return common + ("torch", "transformers", "qwen_vl_utils")
    if model_family == "llava_onevision":
        return common + ("torch", "transformers")
    return common


def _provider_options(config: DeploymentConfig) -> dict[str, object]:
    options = dict(config.provider_options)
    if config.cache_dir is not None and "cache_dir" not in options:
        options["cache_dir"] = str(config.cache_dir)
    return options


def _audit_logger(config: DeploymentConfig) -> PrivacySafeAuditLogger | None:
    if config.audit.path is None:
        return None
    session_id = config.evidence_session_id
    audit_config = config.audit
    if config.evidence_session_mode == "runtime":
        session_id = str(uuid.uuid4())
        template = str(config.audit.path)
        if "{evidence_session_id}" not in template:
            raise ValueError(
                "Runtime evidence sessions require {evidence_session_id} in audit.path."
            )
        audit_config = AuditLogConfig(
            path=Path(template.replace("{evidence_session_id}", session_id)),
            max_bytes=config.audit.max_bytes,
            backup_count=config.audit.backup_count,
            fsync=config.audit.fsync,
        )
    return PrivacySafeAuditLogger(
        audit_config,
        context={
            "deployment_name": config.name,
            "target_profile": config.target_profile,
            "evidence_session_id": session_id,
            "deployment_config_sha256": sha256_file(config.config_path),
            "detector_sha256": sha256_file(config.detector_path),
        },
    )


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _validate_evidence_session_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("evidence_session_id must be a canonical UUID string.")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError("evidence_session_id must be a canonical UUID string.") from exc
    canonical = str(parsed)
    if value != canonical:
        raise ValueError("evidence_session_id must use canonical lowercase hyphenated UUID form.")
    return canonical


def _validate_evidence_session_mode(value: object) -> str:
    if not isinstance(value, str) or value not in {"fixed", "runtime"}:
        raise ValueError("evidence_session_mode must be 'fixed' or 'runtime'.")
    return value


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object.")
    return dict(value)


def _require_keys(
    payload: dict,
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required - optional)
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {unknown}")


def _resolve_path(base_dir: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()
