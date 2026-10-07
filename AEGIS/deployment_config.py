from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import uuid

from AEGIS.audit import (
    AuditLogConfig,
    PrivacySafeAuditLogger,
    validate_traffic_mode,
)
from AEGIS.detector_artifact import DetectorArtifact, load_detector_artifact_snapshot
from AEGIS.detector_set import DetectorHeadConfig, OrDetector, load_or_detector
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
from AEGIS.secret_validation import validate_distinct_secrets, validate_secret
from AEGIS.strict_json import strict_json_loads
from AEGIS.system_resources import (
    ResourceRequirements,
    evaluate_resource_requirements,
    inspect_model_cache,
    inspect_system_resources,
    resolve_model_storage_path,
)


DEPLOYMENT_CONFIG_SCHEMA_VERSION = 1
DEPLOYABLE_ERROR_ACTIONS = frozenset({"review", "block"})
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8766
    api_token_env: str = "AEGIS_API_TOKEN"
    admin_token_env: str = "AEGIS_ADMIN_TOKEN"
    max_body_bytes: int = 12 * 1024 * 1024
    max_concurrent_requests: int = 1
    max_http_connections: int = 32
    request_read_timeout_seconds: float = 15.0
    worker_mode: str = "in_process"
    inference_timeout_seconds: float = 60.0
    worker_startup_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("server.host must be a non-empty string.")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 0 <= self.port <= 65_535:
            raise ValueError("server.port must be in [0, 65535].")
        if (
            not isinstance(self.api_token_env, str)
            or _ENVIRONMENT_NAME.fullmatch(self.api_token_env) is None
        ):
            raise ValueError("server.api_token_env must be an environment variable name.")
        if (
            not isinstance(self.admin_token_env, str)
            or _ENVIRONMENT_NAME.fullmatch(self.admin_token_env) is None
        ):
            raise ValueError(
                "server.admin_token_env must be an environment variable name."
            )
        if self.admin_token_env == self.api_token_env:
            raise ValueError(
                "server.admin_token_env must be distinct from server.api_token_env."
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (
                self.max_body_bytes,
                self.max_concurrent_requests,
                self.max_http_connections,
            )
        ):
            raise ValueError("server size and concurrency limits must be positive.")
        if (
            isinstance(self.request_read_timeout_seconds, bool)
            or not isinstance(self.request_read_timeout_seconds, (int, float))
            or not math.isfinite(float(self.request_read_timeout_seconds))
            or float(self.request_read_timeout_seconds) <= 0
        ):
            raise ValueError(
                "server.request_read_timeout_seconds must be positive and finite."
            )
        if self.worker_mode not in {"in_process", "process"}:
            raise ValueError("server.worker_mode must be 'in_process' or 'process'.")
        if self.worker_mode == "process" and self.max_concurrent_requests != 1:
            raise ValueError(
                "server.worker_mode='process' currently requires max_concurrent_requests=1."
            )
        if (
            isinstance(self.inference_timeout_seconds, bool)
            or not isinstance(self.inference_timeout_seconds, (int, float))
            or not math.isfinite(float(self.inference_timeout_seconds))
            or float(self.inference_timeout_seconds) <= 0
        ):
            raise ValueError("server.inference_timeout_seconds must be positive and finite.")
        if (
            isinstance(self.worker_startup_timeout_seconds, bool)
            or not isinstance(self.worker_startup_timeout_seconds, (int, float))
            or not math.isfinite(float(self.worker_startup_timeout_seconds))
            or float(self.worker_startup_timeout_seconds) <= 0
        ):
            raise ValueError(
                "server.worker_startup_timeout_seconds must be positive and finite."
            )


@dataclass(frozen=True)
class DeploymentConfig:
    name: str
    detector_path: Path | None
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
    detector_heads: tuple[DetectorHeadConfig, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.require_model_cache, bool):
            raise ValueError("require_model_cache must be a boolean.")
        if not isinstance(self.require_cuda, bool):
            raise ValueError("require_cuda must be a boolean.")


def load_deployment_config(path: str | Path) -> DeploymentConfig:
    config_path = Path(path).resolve()
    payload = strict_json_loads(config_path.read_text(encoding="utf-8"))
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
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in {DEPLOYMENT_CONFIG_SCHEMA_VERSION, 2}
    ):
        raise ValueError(
            f"Unsupported deployment config schema {schema_version!r}; expected 1 or 2."
        )
    name = _nonempty_string(payload["name"], "deployment name")
    provider_spec = _nonempty_string(payload["provider"], "provider")
    base_dir = _resolve_path(config_path.parent, payload.get("base_dir", "."))
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
            "admin_token_env",
            "max_body_bytes",
            "max_concurrent_requests",
            "max_http_connections",
            "request_read_timeout_seconds",
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
    detector_path: Path | None = None
    detector_heads: tuple[DetectorHeadConfig, ...] = ()
    if schema_version == 1:
        detector_path = _resolve_path(base_dir, payload["detector"])
    else:
        detector_payload = _mapping(payload["detector"], "detector")
        _require_keys(
            detector_payload,
            required={"mode", "heads"},
            optional=set(),
            context="detector",
        )
        if detector_payload["mode"] != "or":
            raise ValueError("deployment detector.mode must be 'or'.")
        raw_heads = detector_payload["heads"]
        if not isinstance(raw_heads, list):
            raise ValueError("deployment detector.heads must be an array.")
        parsed_heads = []
        for index, raw_head in enumerate(raw_heads):
            head = _mapping(raw_head, f"detector.heads[{index}]")
            _require_keys(
                head,
                required={"name", "artifact", "review_threshold"},
                optional=set(),
                context=f"detector.heads[{index}]",
            )
            review = head["review_threshold"]
            if isinstance(review, bool) or not isinstance(review, (int, float)) or not math.isfinite(float(review)):
                raise ValueError(f"detector.heads[{index}].review_threshold must be finite.")
            parsed_heads.append(
                DetectorHeadConfig(
                    name=_nonempty_string(head["name"], f"detector.heads[{index}].name"),
                    artifact_path=_resolve_path(base_dir, head["artifact"]),
                    review_threshold=float(review),
                )
            )
        detector_heads = tuple(parsed_heads)
        if any(
            value is not None
            for value in (
                policy.block_threshold,
                policy.review_threshold,
                policy.review_margin,
            )
        ):
            raise ValueError(
                "Dual OR deployments require null global block_threshold, "
                "review_threshold, and review_margin values."
            )
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
    require_model_cache = payload.get("require_model_cache", False)
    require_cuda = payload.get("require_cuda", False)
    if not isinstance(require_model_cache, bool):
        raise ValueError("require_model_cache must be a boolean.")
    if not isinstance(require_cuda, bool):
        raise ValueError("require_cuda must be a boolean.")
    target_profile_value = payload.get("target_profile")
    target_profile = (
        None
        if target_profile_value is None
        else _nonempty_string(target_profile_value, "target_profile")
    )
    traffic_mode = validate_traffic_mode(
        _nonempty_string(payload.get("traffic_mode", "shadow"), "traffic_mode")
    )
    return DeploymentConfig(
        name=name,
        detector_path=detector_path,
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
        require_model_cache=require_model_cache,
        require_cuda=require_cuda,
        target_profile=target_profile,
        traffic_mode=traffic_mode,
        evidence_session_mode=evidence_session_mode,
        evidence_session_id=evidence_session_id,
        audit=AuditLogConfig(
            path=(None if audit_path_value is None else _resolve_path(base_dir, audit_path_value)),
            **audit_payload,
        ),
        base_dir=base_dir,
        resources=ResourceRequirements(**resources_payload),
        detector_heads=detector_heads,
    )


def load_configured_detector(config: DeploymentConfig):
    if config.detector_heads:
        return load_or_detector(config.detector_heads)
    if config.detector_path is None:
        raise ValueError("Deployment has no detector artifact configured.")
    artifact, digest = load_detector_artifact_snapshot(config.detector_path)
    artifact.artifact_sha256 = digest
    return artifact


def _assert_final_target_detector_identity(
    config: DeploymentConfig, artifact: DetectorArtifact | OrDetector
) -> None:
    if config.target_profile is None:
        return
    from AEGIS.target_profiles import get_target_profile

    expected = get_target_profile(config.target_profile).detector_identity_sha256
    actual = (
        artifact.identity_sha256
        if isinstance(artifact, OrDetector)
        else getattr(artifact, "artifact_sha256", None)
    )
    if actual != expected:
        raise ValueError(
            "Detector identity changed after deployment preflight; refusing startup."
        )


def _target_input_modalities(
    config: DeploymentConfig,
) -> tuple[str, ...] | None:
    if config.target_profile is None:
        return None
    from AEGIS.target_profiles import get_target_profile

    return get_target_profile(config.target_profile).intended_modalities


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
    admin_token = os.getenv(config.server.admin_token_env) or None
    require_token_for_bind(config.server.host, api_token)
    warmup_payload = None
    if config.warmup_request_path is not None:
        warmup_payload = strict_json_loads(
            config.warmup_request_path.read_text(encoding="utf-8")
        )
        if not isinstance(warmup_payload, dict):
            raise ValueError("Warmup request JSON must contain an object.")
    input_modalities = _target_input_modalities(config)
    if config.server.worker_mode == "process":
        from AEGIS.isolated_service import ProcessIsolatedGuardrailService

        artifact = load_configured_detector(config)
        _assert_final_target_detector_identity(config, artifact)
        isolated_service = ProcessIsolatedGuardrailService(
            artifact_path=config.detector_path,
            detector_heads=config.detector_heads,
            provider_spec=config.provider_spec,
            provider_options=_provider_options(config),
            policy=config.policy,
            limits=config.request_limits,
            api_token=api_token,
            admin_token=admin_token,
            max_body_bytes=config.server.max_body_bytes,
            max_concurrent_requests=config.server.max_concurrent_requests,
            inference_timeout_seconds=config.server.inference_timeout_seconds,
            worker_startup_timeout_seconds=config.server.worker_startup_timeout_seconds,
            warmup_payload=warmup_payload,
            traffic_mode=config.traffic_mode,
            audit_logger=_audit_logger(config, artifact),
            target_profile=config.target_profile,
            input_modalities=input_modalities,
            artifact_snapshot=artifact,
        )
        return isolated_service, {"warmup": isolated_service.startup_warmup}

    artifact = load_configured_detector(config)
    _assert_final_target_detector_identity(config, artifact)
    provider = load_provider(config.provider_spec, _provider_options(config))
    service = GuardrailHTTPService(
        artifact=artifact,
        provider=provider,
        policy=config.policy,
        limits=config.request_limits,
        api_token=api_token,
        admin_token=admin_token,
        max_body_bytes=config.server.max_body_bytes,
        max_concurrent_requests=config.server.max_concurrent_requests,
        traffic_mode=config.traffic_mode,
        audit_logger=_audit_logger(config, artifact),
        target_profile=config.target_profile,
        input_modalities=input_modalities,
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

    detector_paths = (
        tuple(head.artifact_path for head in config.detector_heads)
        if config.detector_heads
        else (() if config.detector_path is None else (config.detector_path,))
    )
    check(
        "detector_exists",
        bool(detector_paths) and all(path.exists() for path in detector_paths),
        f"paths={[str(path) for path in detector_paths]}",
    )
    artifact = None
    provider = None
    target_profile = None
    cuda_devices: tuple[dict[str, object], ...] = ()
    if detector_paths and all(path.exists() for path in detector_paths):
        try:
            artifact = load_configured_detector(config)
            check(
                "detector_loads",
                True,
                f"mode={'or' if isinstance(artifact, OrDetector) else 'single'}, feature_dim={artifact.feature_dim}",
            )
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

            if target_profile.detector_heads:
                configured_heads = {head.name: head for head in config.detector_heads}
                expected_heads = {head.name: head for head in target_profile.detector_heads}
                path_mismatches = {}
                hash_mismatches = {}
                threshold_mismatches = {}
                for name, expected_head in expected_heads.items():
                    configured_head = configured_heads.get(name)
                    expected_asset = resolve_target_asset(
                        expected_head.detector, source_root=config.base_dir
                    )
                    if configured_head is None or configured_head.artifact_path != expected_asset.path:
                        path_mismatches[name] = True
                        continue
                    if sha256_file(configured_head.artifact_path) != expected_head.detector_sha256:
                        hash_mismatches[name] = True
                    if float(configured_head.review_threshold) != float(expected_head.review_threshold):
                        threshold_mismatches[name] = True
                check(
                    "target_profile_detector_matches",
                    set(configured_heads) == set(expected_heads) and not path_mismatches,
                    f"path_mismatches={path_mismatches!r}",
                )
                check(
                    "target_profile_detector_sha256_matches",
                    not hash_mismatches,
                    f"hash_mismatches={hash_mismatches!r}",
                )
                check(
                    "target_profile_detector_thresholds_match",
                    not threshold_mismatches,
                    f"threshold_mismatches={threshold_mismatches!r}",
                )
                check(
                    "target_profile_detector_identity_matches",
                    isinstance(artifact, OrDetector)
                    and artifact.identity_sha256
                    == target_profile.detector_identity_sha256,
                    (
                        f"configured={getattr(artifact, 'identity_sha256', None)!r}, "
                        f"profile={target_profile.detector_identity_sha256!r}"
                    ),
                )
            else:
                assert target_profile.detector is not None
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
                detector_digest = (
                    sha256_file(config.detector_path)
                    if config.detector_path is not None and config.detector_path.is_file()
                    else None
                )
                check(
                    "target_profile_detector_sha256_matches",
                    detector_digest == target_profile.detector_sha256,
                    (
                        f"configured={detector_digest!r}, "
                        f"profile={target_profile.detector_sha256!r}"
                    ),
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
            profile_options = target_profile.provider_options
            option_mismatches = {
                name: {
                    "configured": config.provider_options.get(name),
                    "profile": profile_options.get(name),
                }
                for name in sorted(set(config.provider_options) | set(profile_options))
                if (
                    name not in config.provider_options
                    or name not in profile_options
                    or config.provider_options[name] != profile_options[name]
                )
            }
            check(
                "target_profile_provider_options_match",
                not option_mismatches,
                f"mismatches={option_mismatches!r}",
            )
            pinned_revision_names = ("model_revision", "tokenizer_revision")
            revision_mismatches = {
                name: {
                    "configured": config.provider_options.get(name),
                    "profile": target_profile.provider_options.get(name),
                }
                for name in pinned_revision_names
                if config.provider_options.get(name)
                != target_profile.provider_options.get(name)
            }
            check(
                "target_profile_revisions_match",
                not revision_mismatches,
                f"mismatches={revision_mismatches!r}",
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
                    free_memory, runtime_total_memory = torch.cuda.mem_get_info(index)
                    detail["devices"].append(
                        {
                            "index": index,
                            "name": str(properties.name),
                            "total_memory_bytes": int(properties.total_memory),
                            "runtime_total_memory_bytes": int(runtime_total_memory),
                            "free_memory_bytes": int(free_memory),
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
        expected_content_sha256=(
            None
            if target_profile is None
            else target_profile.model_source.expected_content_sha256
        ),
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
                warmup_payload = strict_json_loads(
                    config.warmup_request_path.read_text(encoding="utf-8")
                )
                check(
                    "warmup_request_valid_json",
                    isinstance(warmup_payload, dict),
                    "warmup request contains a JSON object",
                )
            except (OSError, UnicodeError, ValueError) as exc:
                check("warmup_request_valid_json", False, str(exc))

    token = os.getenv(config.server.api_token_env) or None
    admin_token = os.getenv(config.server.admin_token_env) or None
    try:
        require_token_for_bind(config.server.host, token)
        check(
            "bind_authentication",
            True,
            f"host={config.server.host}, token_configured={token is not None}",
        )
    except ValueError as exc:
        check("bind_authentication", False, str(exc))
    _, api_secret_problems = validate_secret(
        token,
        name="API token",
        required=config.server.host.strip().lower()
        not in {"127.0.0.1", "localhost", "::1"},
    )
    check(
        "api_token_strength",
        not api_secret_problems,
        "configured secret passes strength validation"
        if not api_secret_problems
        else " ".join(api_secret_problems),
    )
    _, admin_secret_problems = validate_secret(
        admin_token,
        name="Admin token",
        required=False,
    )
    check(
        "admin_token_strength",
        not admin_secret_problems,
        "configured secret passes strength validation"
        if not admin_secret_problems
        else " ".join(admin_secret_problems),
        required=admin_token is not None,
    )
    distinct_secret_problems = validate_distinct_secrets(
        {"API token": token, "Admin token": admin_token}
    )
    check(
        "separate_admin_authentication",
        not distinct_secret_problems,
        (
            f"api_env={config.server.api_token_env}, "
            f"admin_env={config.server.admin_token_env}, "
            f"admin_configured={admin_token is not None}; "
            + (
                "configured secrets are distinct"
                if not distinct_secret_problems
                else " ".join(distinct_secret_problems)
            )
        ),
    )
    check(
        "runtime_admin_authentication_configured",
        admin_token is not None,
        f"env={config.server.admin_token_env}, configured={admin_token is not None}",
        required=False,
    )

    if config.policy.fingerprint_key_env is not None:
        fingerprint_key = os.getenv(config.policy.fingerprint_key_env)
        _, fingerprint_secret_problems = validate_secret(
            fingerprint_key,
            name="Fingerprint HMAC key",
            required=True,
        )
        check(
            "fingerprint_hmac_key",
            not fingerprint_secret_problems,
            (
                f"env={config.policy.fingerprint_key_env}, "
                f"configured={bool(fingerprint_key)}; "
                + (
                    "configured secret passes strength validation"
                    if not fingerprint_secret_problems
                    else " ".join(fingerprint_secret_problems)
                )
            ),
        )
        all_secret_problems = validate_distinct_secrets(
            {
                "API token": token,
                "Admin token": admin_token,
                "Fingerprint HMAC key": fingerprint_key,
            }
        )
        check(
            "deployment_secrets_distinct",
            not all_secret_problems,
            "configured secrets are distinct"
            if not all_secret_problems
            else " ".join(all_secret_problems),
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
    runtime_path_bounded = config.evidence_session_mode != "runtime" or (
        config.audit.path is not None
        and "{evidence_session_id}" not in str(config.audit.path)
    )
    check(
        "runtime_evidence_session_stable_audit_path",
        runtime_path_bounded,
        (
            f"mode={config.evidence_session_mode!r}, path={config.audit.path}, "
            "runtime sessions share one bounded rotating target log"
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

    probe_report = None
    if probe_path is not None and artifact is not None and provider is not None:
        try:
            payload = strict_json_loads(
                Path(probe_path).read_text(encoding="utf-8")
            )
            probe_report = evaluate_request_payload(
                artifact,
                provider,
                payload,
                policy=config.policy,
                limits=config.request_limits,
                include_error_details=True,
                input_modalities=_target_input_modalities(config),
                target_profile=config.target_profile,
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
                "name": "serve",
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
        paths = [str(head.artifact_path) for head in config.detector_heads]
        if config.detector_path is not None:
            paths.append(str(config.detector_path))
        return f"Restore and verify the detector artifact(s): {paths}."
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
        "runtime_evidence_session_stable_audit_path",
        "audit_directory_writable",
    }:
        return "Restore the target's privacy-safe audit/session settings and writable audit path."
    if check_name == "resource_preflight":
        return "Free or provision the RAM, disk, model storage, and VRAM required by the target."
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


def _audit_logger(
    config: DeploymentConfig,
    detector: DetectorArtifact | OrDetector | None = None,
) -> PrivacySafeAuditLogger | None:
    if config.audit.path is None:
        return None
    session_id = config.evidence_session_id
    audit_config = config.audit
    if config.evidence_session_mode == "runtime":
        session_id = str(uuid.uuid4())
        configured_path = str(config.audit.path)
        if "{evidence_session_id}" in configured_path:
            raise ValueError(
                "Runtime evidence sessions require one stable audit.path so rotation "
                "bounds storage across process restarts; the session UUID is recorded "
                "inside every event."
            )
        audit_config = AuditLogConfig(
            path=Path(configured_path),
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
            # Never reload detector paths here: audit identity must describe the
            # exact object being served, not a later filesystem state.
            "detector_sha256": (
                None
                if isinstance(detector, OrDetector)
                else (
                    getattr(detector, "artifact_sha256", None)
                    if detector is not None
                    else (
                        sha256_file(config.detector_path)
                        if config.detector_path is not None
                        else None
                    )
                )
            ),
            "detector_identity_sha256": (
                detector.identity_sha256
                if isinstance(detector, OrDetector)
                else (
                    getattr(detector, "artifact_sha256", None)
                    if detector is not None
                    else (
                        sha256_file(config.detector_path)
                        if config.detector_path is not None
                        else None
                    )
                )
            ),
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


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _resolve_path(base_dir: Path, value: object) -> Path:
    path = Path(_nonempty_string(value, "deployment path"))
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()
