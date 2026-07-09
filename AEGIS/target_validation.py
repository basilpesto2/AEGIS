from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class TargetValidationBundle:
    target_name: str
    model_family: str
    model_id: str
    scope: str
    status: str
    intended_modalities: tuple[str, ...]
    detector_path: str
    policy_path: str | None
    summary_paths: tuple[str, ...]
    provider_contract_path: str | None
    calibration_summary_path: str | None
    monitoring_summary_path: str | None
    manifest_path: str | None
    robustness_summary_paths: tuple[str, ...]
    attack_success_summary_paths: tuple[str, ...]
    gate_report: dict[str, object]
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_target_validation_bundle(
    *,
    target_name: str,
    model_family: str,
    model_id: str,
    scope: str,
    detector_path: str,
    summary_paths: list[str],
    gate_report: dict[str, object],
    intended_modalities: list[str] | tuple[str, ...] | None = None,
    policy_path: str | None = None,
    provider_contract_path: str | None = None,
    calibration_summary_path: str | None = None,
    monitoring_summary_path: str | None = None,
    manifest_path: str | None = None,
    robustness_summary_paths: list[str] | tuple[str, ...] | None = None,
    attack_success_summary_paths: list[str] | tuple[str, ...] | None = None,
    notes: str = "",
) -> TargetValidationBundle:
    status = _bundle_status(
        gate_report=gate_report,
        policy_path=policy_path,
        provider_contract_path=provider_contract_path,
        calibration_summary_path=calibration_summary_path,
        monitoring_summary_path=monitoring_summary_path,
    )
    return TargetValidationBundle(
        target_name=target_name,
        model_family=model_family,
        model_id=model_id,
        scope=scope,
        status=status,
        intended_modalities=tuple(intended_modalities or ()),
        detector_path=detector_path,
        policy_path=policy_path,
        summary_paths=tuple(summary_paths),
        provider_contract_path=provider_contract_path,
        calibration_summary_path=calibration_summary_path,
        monitoring_summary_path=monitoring_summary_path,
        manifest_path=manifest_path,
        robustness_summary_paths=tuple(robustness_summary_paths or ()),
        attack_success_summary_paths=tuple(attack_success_summary_paths or ()),
        gate_report=gate_report,
        notes=notes,
    )


def render_target_bundle_markdown(bundle: TargetValidationBundle) -> str:
    payload = bundle.to_dict()
    lines = [
        f"# Target Validation Bundle: {bundle.target_name}",
        "",
        f"Status: `{bundle.status}`",
        f"Scope: `{bundle.scope}`",
        f"Model family: `{bundle.model_family}`",
        f"Model ID: `{bundle.model_id}`",
        f"Intended modalities: `{', '.join(bundle.intended_modalities) or 'unspecified'}`",
        "",
        "## Artifacts",
        "",
        f"- Detector: `{bundle.detector_path}`",
        f"- Policy: `{bundle.policy_path or 'missing'}`",
        f"- Provider contract: `{bundle.provider_contract_path or 'missing'}`",
        f"- Calibration summary: `{bundle.calibration_summary_path or 'missing'}`",
        f"- Monitoring summary: `{bundle.monitoring_summary_path or 'missing'}`",
        f"- Manifest: `{bundle.manifest_path or 'missing'}`",
        "",
        "## Evaluation Summaries",
        "",
    ]
    for path in bundle.summary_paths:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Robustness Evidence", ""])
    if bundle.robustness_summary_paths:
        for path in bundle.robustness_summary_paths:
            lines.append(f"- `{path}`")
    else:
        lines.append("- `missing`")
    lines.extend(["", "## Attack-Success Evidence", ""])
    if bundle.attack_success_summary_paths:
        for path in bundle.attack_success_summary_paths:
            lines.append(f"- `{path}`")
    else:
        lines.append("- `missing`")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Gate OK: `{payload['gate_report'].get('ok')}`",
            f"- Coverage: `{payload['gate_report'].get('coverage', {})}`",
            "",
            "## Notes",
            "",
            bundle.notes or "None.",
            "",
        ]
    )
    return "\n".join(lines)


def update_registry(
    registry_path: str | Path,
    bundle: TargetValidationBundle,
) -> dict[str, object]:
    path = Path(registry_path)
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    else:
        registry = {"schema_version": 1, "targets": []}
    targets = [
        target
        for target in registry.get("targets", [])
        if target.get("target_name") != bundle.target_name
    ]
    targets.append(bundle.to_dict())
    targets.sort(key=lambda item: str(item.get("target_name", "")))
    registry["targets"] = targets
    registry["status_counts"] = _status_counts(targets)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    return registry


def _bundle_status(
    *,
    gate_report: dict[str, object],
    policy_path: str | None,
    provider_contract_path: str | None,
    calibration_summary_path: str | None,
    monitoring_summary_path: str | None,
) -> str:
    if not bool(gate_report.get("ok")):
        return "failed_gate"
    missing = [
        name
        for name, value in {
            "policy": policy_path,
            "provider_contract": provider_contract_path,
            "calibration": calibration_summary_path,
            "monitoring": monitoring_summary_path,
        }.items()
        if not value
    ]
    if missing:
        return "controlled_evidence_only"
    artifact_paths = {
        "policy": policy_path,
        "provider_contract": provider_contract_path,
        "calibration": calibration_summary_path,
        "monitoring": monitoring_summary_path,
    }
    missing_files = [
        name
        for name, value in artifact_paths.items()
        if value is None or not Path(value).exists()
    ]
    if missing_files:
        return "controlled_evidence_only"
    if not _json_bool(provider_contract_path, "ok"):
        return "failed_provider_contract"
    if not _json_bool(calibration_summary_path, "feasible"):
        return "failed_calibration"
    if not _json_positive_int(monitoring_summary_path, "n_samples"):
        return "failed_monitoring"
    return "production_candidate"


def _status_counts(targets: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in targets:
        status = str(target.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _json_bool(path: str | Path | None, key: str) -> bool:
    if path is None:
        return False
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get(key))


def _json_positive_int(path: str | Path | None, key: str) -> bool:
    if path is None:
        return False
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        return int(payload.get(key, 0)) > 0
    except (TypeError, ValueError):
        return False
