from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Any

from AEGIS.readiness import sha256_file
from AEGIS.target_release import validate_target_release


DEPLOYMENT_PACKAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DeploymentPackageCheck:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_target_deployment_package(
    bundle_path: str | Path,
    output_dir: str | Path,
    *,
    root: str | Path = ".",
    require_labeled_monitoring: bool = False,
    require_robustness_evidence: bool = False,
    require_attack_success_evidence: bool = False,
    required_provider_modalities: tuple[str, ...] = (),
) -> dict[str, object]:
    """Create a portable deployment directory from a validated target bundle."""

    root_path = Path(root)
    output_path = Path(output_dir)
    source_release_report = validate_target_release(
        bundle_path,
        root=root_path,
        require_labeled_monitoring=require_labeled_monitoring,
        require_robustness_evidence=require_robustness_evidence,
        require_attack_success_evidence=require_attack_success_evidence,
        required_provider_modalities=required_provider_modalities,
    )
    if not bool(source_release_report.get("ok")):
        failed = [
            str(item.get("name"))
            for item in source_release_report.get("checks", [])
            if isinstance(item, dict) and not bool(item.get("ok"))
        ]
        raise ValueError(f"Source target release failed validation: {failed}")

    source_bundle_file = _resolve(root_path, bundle_path)
    source_bundle = _read_json(source_bundle_file)
    output_path.mkdir(parents=True, exist_ok=True)

    packaged_paths = _copy_release_artifacts(source_bundle, root_path, output_path)
    _rewrite_packaged_policy(
        output_path / packaged_paths["policy"],
        detector_path=packaged_paths["detector"],
    )
    packaged_bundle = dict(source_bundle)
    packaged_bundle["detector_path"] = packaged_paths["detector"]
    packaged_bundle["policy_path"] = packaged_paths["policy"]
    packaged_bundle["provider_contract_path"] = packaged_paths["provider_contract"]
    packaged_bundle["calibration_summary_path"] = packaged_paths["calibration"]
    packaged_bundle["monitoring_summary_path"] = packaged_paths["monitoring"]
    packaged_bundle["summary_paths"] = packaged_paths["summaries"]
    packaged_bundle["robustness_summary_paths"] = packaged_paths["robustness"]
    packaged_bundle["attack_success_summary_paths"] = packaged_paths["attack_success"]
    packaged_bundle["manifest_path"] = "deployment_manifest.json"

    bundle_output = output_path / "bundle.json"
    bundle_output.write_text(
        json.dumps(packaged_bundle, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    artifact_records = _artifact_records(
        output_path,
        package_paths=[
            packaged_paths["detector"],
            packaged_paths["policy"],
            packaged_paths["provider_contract"],
            packaged_paths["calibration"],
            packaged_paths["monitoring"],
            *packaged_paths["summaries"],
            *packaged_paths["robustness"],
            *packaged_paths["attack_success"],
            "bundle.json",
        ],
        source_paths={
            packaged_paths["detector"]: str(source_bundle.get("detector_path")),
            packaged_paths["policy"]: str(source_bundle.get("policy_path")),
            packaged_paths["provider_contract"]: str(source_bundle.get("provider_contract_path")),
            packaged_paths["calibration"]: str(source_bundle.get("calibration_summary_path")),
            packaged_paths["monitoring"]: str(source_bundle.get("monitoring_summary_path")),
            "bundle.json": str(bundle_path),
            **{
                package_path: source_path
                for package_path, source_path in zip(
                    packaged_paths["robustness"],
                    source_bundle.get("robustness_summary_paths", []) or [],
                    strict=False,
                )
            },
            **{
                package_path: source_path
                for package_path, source_path in zip(
                    packaged_paths["attack_success"],
                    source_bundle.get("attack_success_summary_paths", []) or [],
                    strict=False,
                )
            },
        },
    )
    if source_bundle.get("manifest_path"):
        source_manifest_path = _resolve(root_path, str(source_bundle["manifest_path"]))
        if source_manifest_path.exists():
            package_manifest_source = output_path / "source_manifest.json"
            shutil.copy2(source_manifest_path, package_manifest_source)
            artifact_records.append(
                _describe_package_artifact(
                    output_path,
                    "source_manifest.json",
                    source_path=str(source_bundle["manifest_path"]),
                    role="source_manifest",
                )
            )

    deployment_manifest = {
        "schema_version": DEPLOYMENT_PACKAGE_SCHEMA_VERSION,
        "target": {
            "target_name": packaged_bundle.get("target_name"),
            "model_family": packaged_bundle.get("model_family"),
            "model_id": packaged_bundle.get("model_id"),
            "scope": packaged_bundle.get("scope"),
            "status": packaged_bundle.get("status"),
            "intended_modalities": packaged_bundle.get("intended_modalities", []),
        },
        "source_bundle_path": str(bundle_path),
        "artifacts": artifact_records,
        "safety_note": (
            "Package manifests contain hashes, paths, and aggregate evidence only. "
            "Raw prompt text and generated harmful responses are not required for deployment."
        ),
    }
    (output_path / "deployment_manifest.json").write_text(
        json.dumps(deployment_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    package_release_report = validate_target_release(
        "bundle.json",
        root=output_path,
        require_labeled_monitoring=require_labeled_monitoring,
        require_robustness_evidence=require_robustness_evidence,
        require_attack_success_evidence=require_attack_success_evidence,
        required_provider_modalities=required_provider_modalities,
    )
    package_report = {
        "ok": bool(package_release_report.get("ok")),
        "package_dir": str(output_path),
        "bundle_path": "bundle.json",
        "manifest_path": "deployment_manifest.json",
        "target": deployment_manifest["target"],
        "artifacts": artifact_records,
        "source_release_report": source_release_report,
        "package_release_report": package_release_report,
    }
    (output_path / "deployment_package_report.json").write_text(
        json.dumps(package_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not bool(package_release_report.get("ok")):
        raise ValueError("Packaged target release failed validation.")
    return package_report


def validate_target_deployment_package(
    package_dir: str | Path,
    *,
    require_labeled_monitoring: bool = False,
    require_robustness_evidence: bool = False,
    require_attack_success_evidence: bool = False,
    required_provider_modalities: tuple[str, ...] = (),
) -> dict[str, object]:
    package_path = Path(package_dir)
    checks: list[DeploymentPackageCheck] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append(DeploymentPackageCheck(name=name, ok=bool(ok), detail=detail))

    manifest_path = package_path / "deployment_manifest.json"
    bundle_path = package_path / "bundle.json"
    check("package_dir_exists", package_path.exists() and package_path.is_dir(), str(package_path))
    check("deployment_manifest_exists", manifest_path.exists(), str(manifest_path))
    check("bundle_exists", bundle_path.exists(), str(bundle_path))

    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    check(
        "deployment_manifest_schema",
        manifest.get("schema_version") == DEPLOYMENT_PACKAGE_SCHEMA_VERSION,
        f"schema_version={manifest.get('schema_version')!r}",
    )

    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        check("deployment_manifest_artifacts_list", False, "artifacts is not a list")
        artifacts = []
    for idx, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            check(f"artifact_{idx}_record", False, "artifact record is not a dict")
            continue
        package_rel = str(artifact.get("path", ""))
        artifact_path = package_path / package_rel
        exists = artifact_path.exists()
        check(f"artifact_exists:{package_rel}", exists, str(artifact_path))
        if exists and artifact.get("sha256"):
            actual = sha256_file(artifact_path)
            check(
                f"artifact_sha256:{package_rel}",
                actual == artifact.get("sha256"),
                f"expected={artifact.get('sha256')}, actual={actual}",
            )

    release_report = validate_target_release(
        "bundle.json",
        root=package_path,
        require_labeled_monitoring=require_labeled_monitoring,
        require_robustness_evidence=require_robustness_evidence,
        require_attack_success_evidence=require_attack_success_evidence,
        required_provider_modalities=required_provider_modalities,
    )
    check(
        "packaged_target_release_ok",
        bool(release_report.get("ok")),
        f"ok={release_report.get('ok')!r}",
    )
    return {
        "ok": all(item.ok for item in checks),
        "package_dir": str(package_path),
        "target": manifest.get("target", {}),
        "checks": [item.to_dict() for item in checks],
        "release_report": release_report,
    }


def _copy_release_artifacts(
    bundle: dict[str, Any],
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "detector": _copy_one(root, output_dir, bundle["detector_path"], "artifacts/detector"),
        "policy": _copy_one(root, output_dir, bundle["policy_path"], "artifacts/policy"),
        "provider_contract": _copy_one(
            root,
            output_dir,
            bundle["provider_contract_path"],
            "artifacts/provider_contract",
        ),
        "calibration": _copy_one(
            root,
            output_dir,
            bundle["calibration_summary_path"],
            "artifacts/calibration",
        ),
        "monitoring": _copy_one(
            root,
            output_dir,
            bundle["monitoring_summary_path"],
            "artifacts/monitoring",
        ),
    }
    summaries = []
    for idx, summary_path in enumerate(bundle.get("summary_paths", [])):
        summaries.append(
            _copy_one(
                root,
                output_dir,
                summary_path,
                f"artifacts/summaries/{idx:02d}",
            )
        )
    mapping["summaries"] = summaries
    robustness = []
    for idx, path in enumerate(bundle.get("robustness_summary_paths", [])):
        robustness.append(_copy_one(root, output_dir, path, f"artifacts/robustness/{idx:02d}"))
    mapping["robustness"] = robustness
    attack_success = []
    for idx, path in enumerate(bundle.get("attack_success_summary_paths", [])):
        attack_success.append(
            _copy_one(root, output_dir, path, f"artifacts/attack_success/{idx:02d}")
        )
    mapping["attack_success"] = attack_success
    return mapping


def _copy_one(root: Path, output_dir: Path, source: str, package_prefix: str) -> str:
    source_path = _resolve(root, source)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Deployment artifact is missing: {source_path}")
    relative_output = Path(package_prefix) / source_path.name
    target_path = output_dir / relative_output
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return _as_posix(relative_output)


def _artifact_records(
    package_root: Path,
    *,
    package_paths: list[str],
    source_paths: dict[str, str],
) -> list[dict[str, object]]:
    records = []
    for package_path in package_paths:
        records.append(
            _describe_package_artifact(
                package_root,
                package_path,
                source_path=source_paths.get(package_path),
                role=_artifact_role(package_path),
            )
        )
    return records


def _rewrite_packaged_policy(policy_path: Path, *, detector_path: str) -> None:
    policy = _read_json(policy_path)
    if not policy:
        return
    policy["detector"] = detector_path
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")


def _describe_package_artifact(
    package_root: Path,
    package_path: str,
    *,
    source_path: str | None,
    role: str,
) -> dict[str, object]:
    path = package_root / package_path
    return {
        "path": _as_posix(Path(package_path)),
        "role": role,
        "source_path": source_path,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _artifact_role(package_path: str) -> str:
    path = _as_posix(Path(package_path))
    if path == "bundle.json":
        return "target_bundle"
    parts = path.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "artifacts" else "artifact"


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _as_posix(path: Path) -> str:
    return path.as_posix()
