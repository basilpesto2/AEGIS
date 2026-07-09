from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from AEGIS.detector_artifact import load_detector_artifact


@dataclass(frozen=True)
class TargetReleaseCheck:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_target_release(
    bundle_path: str | Path,
    *,
    root: str | Path = ".",
    require_status: str = "production_candidate",
    require_labeled_monitoring: bool = False,
    require_robustness_evidence: bool = False,
    require_attack_success_evidence: bool = False,
    required_provider_modalities: tuple[str, ...] = (),
) -> dict[str, object]:
    root_path = Path(root)
    bundle_file = _resolve(root_path, bundle_path)
    checks: list[TargetReleaseCheck] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append(TargetReleaseCheck(name, bool(ok), detail))

    check("bundle_exists", bundle_file.exists(), f"path={bundle_file}")
    if not bundle_file.exists():
        return _report(False, checks, {"path": str(bundle_file)})

    bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
    check(
        "bundle_status",
        bundle.get("status") == require_status,
        f"status={bundle.get('status')!r}, required={require_status!r}",
    )
    intended_modalities = _bundle_intended_modalities(bundle)
    invalid_modalities = sorted(set(intended_modalities) - _VALID_MODALITIES)
    if "intended_modalities" in bundle:
        check(
            "intended_modalities_valid",
            not invalid_modalities,
            f"intended_modalities={intended_modalities}, invalid={invalid_modalities}",
        )
    gate_report = _dict(bundle.get("gate_report"))
    check("gate_report_ok", bool(gate_report.get("ok")), f"ok={gate_report.get('ok')!r}")
    gate_checks = gate_report.get("checks", [])
    if isinstance(gate_checks, list):
        failed_gate_checks = [
            str(item.get("name", idx))
            for idx, item in enumerate(gate_checks)
            if isinstance(item, dict) and not bool(item.get("ok"))
        ]
    else:
        failed_gate_checks = ["gate_report.checks is not a list"]
    check("gate_checks_ok", not failed_gate_checks, f"failed={failed_gate_checks}")

    path_fields = {
        "detector": bundle.get("detector_path"),
        "policy": bundle.get("policy_path"),
        "provider_contract": bundle.get("provider_contract_path"),
        "calibration": bundle.get("calibration_summary_path"),
        "monitoring": bundle.get("monitoring_summary_path"),
        "manifest": bundle.get("manifest_path"),
    }
    resolved_paths = {
        name: _resolve_optional(root_path, value)
        for name, value in path_fields.items()
    }
    for name, path in resolved_paths.items():
        check(f"{name}_exists", path is not None and path.exists(), f"path={path}")

    summary_paths = [
        _resolve(root_path, path)
        for path in bundle.get("summary_paths", [])
        if isinstance(path, str)
    ]
    check("has_summary_paths", bool(summary_paths), f"count={len(summary_paths)}")
    missing_summaries = [str(path) for path in summary_paths if not path.exists()]
    check("summary_paths_exist", not missing_summaries, f"missing={missing_summaries}")
    robustness_paths = _bundle_path_list(root_path, bundle, "robustness_summary_paths")
    attack_success_paths = _bundle_path_list(root_path, bundle, "attack_success_summary_paths")
    _check_path_list_evidence(
        check,
        "robustness_evidence",
        robustness_paths,
        require=require_robustness_evidence,
    )
    _check_path_list_evidence(
        check,
        "attack_success_evidence",
        attack_success_paths,
        require=require_attack_success_evidence,
    )

    artifact = None
    detector_path = resolved_paths["detector"]
    if detector_path is not None and detector_path.exists():
        try:
            artifact = load_detector_artifact(detector_path)
            check("detector_loads", True, f"feature_dim={artifact.feature_dim}")
            check(
                "detector_model_family_matches_bundle",
                artifact.model_family == bundle.get("model_family"),
                f"detector={artifact.model_family!r}, bundle={bundle.get('model_family')!r}",
            )
            check(
                "detector_model_id_matches_bundle",
                artifact.model_id == bundle.get("model_id"),
                f"detector={artifact.model_id!r}, bundle={bundle.get('model_id')!r}",
            )
        except Exception as exc:
            check("detector_loads", False, str(exc))

    policy = _read_json(resolved_paths["policy"])
    provider_contract = _read_json(resolved_paths["provider_contract"])
    calibration = _read_json(resolved_paths["calibration"])
    monitoring = _read_json(resolved_paths["monitoring"])
    manifest = _read_json(resolved_paths["manifest"])

    _check_provider_contract(
        check,
        provider_contract,
        bundle,
        required_provider_modalities=_merge_modalities(
            intended_modalities,
            required_provider_modalities,
        ),
    )
    _check_calibration_policy(check, calibration, policy, bundle)
    _check_monitoring(check, monitoring, require_labeled_monitoring=require_labeled_monitoring)
    _check_summary_provenance(check, summary_paths, bundle)
    _check_manifest_membership(
        check,
        manifest,
        expected_paths=[
            str(value)
            for name, value in path_fields.items()
            if name != "manifest" and isinstance(value, str) and value
        ]
        + [
            str(path)
            for path in bundle.get("summary_paths", [])
            if isinstance(path, str)
        ]
        + [
            str(path)
            for path in bundle.get("robustness_summary_paths", [])
            if isinstance(path, str)
        ]
        + [
            str(path)
            for path in bundle.get("attack_success_summary_paths", [])
            if isinstance(path, str)
        ]
        + [str(bundle_path)],
    )

    return _report(all(item.ok for item in checks), checks, bundle)


def validate_target_registry(
    registry_path: str | Path,
    *,
    root: str | Path = ".",
    min_production_candidates: int = 1,
    min_model_families: int = 1,
    required_model_families: tuple[str, ...] = (),
    required_target_modalities: tuple[str, ...] = (),
    require_labeled_monitoring: bool = False,
    require_robustness_evidence: bool = False,
    require_attack_success_evidence: bool = False,
    required_provider_modalities: tuple[str, ...] = (),
) -> dict[str, object]:
    root_path = Path(root)
    registry_file = _resolve(root_path, registry_path)
    checks: list[TargetReleaseCheck] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append(TargetReleaseCheck(name, bool(ok), detail))

    check("registry_exists", registry_file.exists(), f"path={registry_file}")
    if not registry_file.exists():
        return _registry_report(False, checks, {}, [])

    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    targets = registry.get("targets", [])
    check("registry_has_targets", isinstance(targets, list) and bool(targets), f"count={len(targets) if isinstance(targets, list) else 'invalid'}")
    if not isinstance(targets, list):
        return _registry_report(False, checks, registry, [])

    names = [str(target.get("target_name", "")) for target in targets if isinstance(target, dict)]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    check("target_names_unique", not duplicate_names, f"duplicates={duplicate_names}")

    actual_status_counts: dict[str, int] = {}
    for target in targets:
        if isinstance(target, dict):
            status = str(target.get("status", "unknown"))
            actual_status_counts[status] = actual_status_counts.get(status, 0) + 1
    check(
        "status_counts_match_targets",
        registry.get("status_counts") == actual_status_counts,
        f"registry={registry.get('status_counts')!r}, actual={actual_status_counts!r}",
    )

    production_targets = [
        target
        for target in targets
        if isinstance(target, dict) and target.get("status") == "production_candidate"
    ]
    families = sorted({str(target.get("model_family")) for target in production_targets})
    target_modalities = sorted(
        {
            modality
            for target in production_targets
            for modality in _bundle_intended_modalities(target)
        }
    )
    invalid_target_modalities = sorted(set(target_modalities) - _VALID_MODALITIES)
    check(
        "min_production_candidates",
        len(production_targets) >= min_production_candidates,
        f"production_candidates={len(production_targets)}, required>={min_production_candidates}",
    )
    check(
        "min_model_families",
        len(families) >= min_model_families,
        f"model_families={families}, required>={min_model_families}",
    )
    missing_families = sorted(set(required_model_families) - set(families))
    check("required_model_families", not missing_families, f"missing={missing_families}")
    check(
        "registry_target_modalities_valid",
        not invalid_target_modalities,
        f"target_modalities={target_modalities}, invalid={invalid_target_modalities}",
    )
    missing_modalities = sorted(set(required_target_modalities) - set(target_modalities))
    check(
        "required_target_modalities",
        not missing_modalities,
        f"target_modalities={target_modalities}, missing={missing_modalities}",
    )

    target_reports = []
    for target in production_targets:
        bundle_path = _registry_bundle_path(target)
        report = validate_target_release(
            bundle_path,
            root=root_path,
            require_status="production_candidate",
            require_labeled_monitoring=require_labeled_monitoring,
            require_robustness_evidence=require_robustness_evidence,
            require_attack_success_evidence=require_attack_success_evidence,
            required_provider_modalities=required_provider_modalities,
        )
        target_reports.append({"bundle_path": bundle_path, "report": report})
        check(
            f"target_release_ok:{target.get('target_name')}",
            bool(report.get("ok")),
            f"bundle={bundle_path}",
        )

    return _registry_report(all(item.ok for item in checks), checks, registry, target_reports)


def _check_provider_contract(
    check,
    provider_contract: dict[str, Any],
    bundle: dict[str, Any],
    *,
    required_provider_modalities: tuple[str, ...],
) -> None:
    check("provider_contract_ok", bool(provider_contract.get("ok")), f"ok={provider_contract.get('ok')!r}")
    provider = _dict(provider_contract.get("provider"))
    detector = _dict(provider_contract.get("detector"))
    for source_name, payload in {"provider": provider, "detector": detector}.items():
        check(
            f"{source_name}_model_family_matches_bundle",
            payload.get("model_family") == bundle.get("model_family"),
            f"{source_name}={payload.get('model_family')!r}, bundle={bundle.get('model_family')!r}",
        )
        check(
            f"{source_name}_model_id_matches_bundle",
            payload.get("model_id") == bundle.get("model_id"),
            f"{source_name}={payload.get('model_id')!r}, bundle={bundle.get('model_id')!r}",
        )
    check(
        "provider_feature_dim_matches_detector",
        provider.get("feature_dim") == detector.get("feature_dim"),
        f"provider={provider.get('feature_dim')!r}, detector={detector.get('feature_dim')!r}",
    )
    if required_provider_modalities:
        coverage = _dict(provider_contract.get("request_coverage"))
        modalities = coverage.get("modalities", [])
        if not isinstance(modalities, list):
            modalities = []
        observed = sorted(str(value) for value in modalities)
        check(
            "provider_contract_has_request_coverage",
            bool(coverage),
            f"coverage_keys={sorted(coverage) if coverage else []}",
        )
        for modality in required_provider_modalities:
            check(
                f"provider_contract_covers_modality:{modality}",
                modality in observed,
                f"observed_modalities={observed}",
            )


def _check_calibration_policy(
    check,
    calibration: dict[str, Any],
    policy: dict[str, Any],
    bundle: dict[str, Any],
) -> None:
    check("calibration_feasible", bool(calibration.get("feasible")), f"feasible={calibration.get('feasible')!r}")
    check("calibration_has_samples", int(calibration.get("n_samples", 0)) > 0, f"n_samples={calibration.get('n_samples')!r}")
    policy_config = _dict(policy.get("policy"))
    check(
        "policy_detector_matches_bundle",
        _same_path_string(policy.get("detector"), bundle.get("detector_path")),
        f"policy={policy.get('detector')!r}, bundle={bundle.get('detector_path')!r}",
    )
    check(
        "policy_requires_matching_provenance",
        bool(policy_config.get("require_matching_provenance")),
        f"require_matching_provenance={policy_config.get('require_matching_provenance')!r}",
    )
    check(
        "policy_action_on_error_review",
        policy_config.get("action_on_error") == "review",
        f"action_on_error={policy_config.get('action_on_error')!r}",
    )
    check(
        "policy_threshold_matches_calibration",
        _same_float(policy_config.get("block_threshold"), calibration.get("threshold")),
        (
            f"policy={policy_config.get('block_threshold')!r}, "
            f"calibration={calibration.get('threshold')!r}"
        ),
    )
    check(
        "policy_review_margin_matches_calibration",
        _same_float(policy_config.get("review_margin"), calibration.get("review_margin")),
        (
            f"policy={policy_config.get('review_margin')!r}, "
            f"calibration={calibration.get('review_margin')!r}"
        ),
    )


def _check_monitoring(
    check,
    monitoring: dict[str, Any],
    *,
    require_labeled_monitoring: bool,
) -> None:
    check("monitoring_has_samples", int(monitoring.get("n_samples", 0)) > 0, f"n_samples={monitoring.get('n_samples')!r}")
    check(
        "monitoring_has_actions",
        bool(_dict(monitoring.get("action_counts"))),
        f"action_counts={monitoring.get('action_counts')!r}",
    )
    if require_labeled_monitoring:
        check(
            "monitoring_has_labeled_metrics",
            bool(_dict(monitoring.get("labeled_metrics"))),
            f"labeled_metrics={monitoring.get('labeled_metrics')!r}",
        )


def _check_path_list_evidence(
    check,
    name: str,
    paths: list[Path],
    *,
    require: bool,
) -> None:
    if require:
        check(f"{name}_present", bool(paths), f"count={len(paths)}")
    missing = [str(path) for path in paths if not path.exists()]
    if paths or require:
        check(f"{name}_paths_exist", not missing, f"missing={missing}")


def _check_summary_provenance(
    check,
    summary_paths: list[Path],
    bundle: dict[str, Any],
) -> None:
    for path in summary_paths:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            check(f"summary_readable:{path.name}", False, str(exc))
            continue
        detector = _dict(summary.get("detector"))
        check(
            f"summary_model_family_matches_bundle:{path.name}",
            detector.get("model_family") == bundle.get("model_family"),
            f"summary={detector.get('model_family')!r}, bundle={bundle.get('model_family')!r}",
        )
        check(
            f"summary_model_id_matches_bundle:{path.name}",
            detector.get("model_id") == bundle.get("model_id"),
            f"summary={detector.get('model_id')!r}, bundle={bundle.get('model_id')!r}",
        )


def _check_manifest_membership(
    check,
    manifest: dict[str, Any],
    *,
    expected_paths: list[str],
) -> None:
    artifacts = manifest.get("artifacts", [])
    manifest_paths = set()
    if isinstance(artifacts, list):
        for item in artifacts:
            if isinstance(item, dict) and item.get("path"):
                manifest_paths.add(_normalize_path(str(item["path"])))
    else:
        check("manifest_artifacts_list", False, "manifest artifacts is not a list")
        return

    expected = [_normalize_path(path) for path in expected_paths]
    missing = [
        path
        for path in expected
        if path not in manifest_paths and _normalize_path(_relative_or_self(path)) not in manifest_paths
    ]
    check("manifest_contains_release_artifacts", not missing, f"missing={missing}")


def _report(ok: bool, checks: list[TargetReleaseCheck], bundle: dict[str, Any]) -> dict[str, object]:
    return {
        "ok": bool(ok),
        "bundle": {
            "target_name": bundle.get("target_name"),
            "status": bundle.get("status"),
            "model_family": bundle.get("model_family"),
            "model_id": bundle.get("model_id"),
            "scope": bundle.get("scope"),
        },
        "checks": [item.to_dict() for item in checks],
    }


def _registry_report(
    ok: bool,
    checks: list[TargetReleaseCheck],
    registry: dict[str, Any],
    target_reports: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "ok": bool(ok),
        "registry": {
            "schema_version": registry.get("schema_version"),
            "status_counts": registry.get("status_counts"),
            "n_targets": len(registry.get("targets", [])) if isinstance(registry.get("targets"), list) else 0,
        },
        "coverage": _registry_coverage(registry),
        "checks": [item.to_dict() for item in checks],
        "target_reports": target_reports,
    }


def _registry_bundle_path(target: dict[str, Any]) -> str:
    if target.get("bundle_path"):
        return str(target["bundle_path"])
    target_name = str(target.get("target_name", ""))
    return str(Path("docs") / "targets" / f"{target_name}.json")


def _registry_coverage(registry: dict[str, Any]) -> dict[str, object]:
    targets = registry.get("targets", [])
    if not isinstance(targets, list):
        return {
            "production_target_names": [],
            "model_families": [],
            "target_modalities": [],
        }
    production_targets = [
        target
        for target in targets
        if isinstance(target, dict) and target.get("status") == "production_candidate"
    ]
    return {
        "production_target_names": sorted(
            str(target.get("target_name"))
            for target in production_targets
            if target.get("target_name")
        ),
        "model_families": sorted(
            str(target.get("model_family"))
            for target in production_targets
            if target.get("model_family")
        ),
        "target_modalities": sorted(
            {
                modality
                for target in production_targets
                for modality in _bundle_intended_modalities(target)
            }
        ),
    }


_VALID_MODALITIES = {"text", "image", "image_text"}


def _bundle_intended_modalities(bundle: dict[str, Any]) -> tuple[str, ...]:
    modalities = bundle.get("intended_modalities", [])
    if isinstance(modalities, str):
        return (modalities,)
    if not isinstance(modalities, list):
        return ()
    return tuple(str(value) for value in modalities)


def _merge_modalities(
    bundle_modalities: tuple[str, ...],
    requested_modalities: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(sorted({*bundle_modalities, *requested_modalities}))


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _bundle_path_list(root: Path, bundle: dict[str, Any], field: str) -> list[Path]:
    values = bundle.get(field, [])
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return [_resolve(root, value) for value in values if isinstance(value, str)]


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _resolve_optional(root: Path, path: object) -> Path | None:
    if not path:
        return None
    return _resolve(root, str(path))


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _same_float(left: object, right: object, tolerance: float = 1e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _same_path_string(left: object, right: object) -> bool:
    if left is None or right is None:
        return False
    return _normalize_path(str(left)) == _normalize_path(str(right))


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip("./")


def _relative_or_self(path: str) -> str:
    marker = "FYP/"
    normalized = _normalize_path(path)
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return normalized
