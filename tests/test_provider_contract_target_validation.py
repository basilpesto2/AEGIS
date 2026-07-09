from __future__ import annotations

import numpy as np

from AEGIS.deployment_package import (
    build_target_deployment_package,
    validate_target_deployment_package,
)
from AEGIS.detector_artifact import DetectorArtifact, save_detector_artifact
from AEGIS.guardrail import GuardrailRequest
from AEGIS.logistic import LogisticRegressionNumpy
from AEGIS.provider_contract import validate_provider_contract
from AEGIS.providers import LlavaOnevisionGuardrailProvider, Qwen25VLGuardrailProvider
from AEGIS.target_release import validate_target_registry, validate_target_release
from AEGIS.target_validation import (
    build_target_validation_bundle,
    render_target_bundle_markdown,
    update_registry,
)


class UnitProvider:
    model_family = "generic"
    model_id = "unit-model"
    pooling = "text_tokens"
    feature_dim = 2

    def embed(self, request: GuardrailRequest) -> np.ndarray:
        return np.asarray([1.0, 0.0])


def _artifact() -> DetectorArtifact:
    classifier = LogisticRegressionNumpy(standardize=False)
    classifier.weights_ = np.asarray([4.0, 0.0])
    classifier.bias_ = -2.0
    classifier.mean_ = np.asarray([0.0, 0.0])
    classifier.scale_ = np.asarray([1.0, 1.0])
    return DetectorArtifact(
        classifier=classifier,
        threshold=0.5,
        model_family="generic",
        model_id="unit-model",
        layer=-1,
        pooling="text_tokens",
        source="unit",
    )


def test_provider_contract_accepts_matching_provider() -> None:
    report = validate_provider_contract(
        UnitProvider(),
        _artifact(),
        requests=[GuardrailRequest(text="hello", request_id="r0")],
    )

    assert report["ok"] is True
    assert report["provider"]["model_id"] == "unit-model"
    assert report["request_coverage"]["modalities"] == ["text"]


def test_provider_contract_enforces_required_modality() -> None:
    report = validate_provider_contract(
        UnitProvider(),
        _artifact(),
        requests=[GuardrailRequest(text="hello", request_id="r0")],
        required_modalities=("image_text",),
    )

    assert report["ok"] is False
    failed = {item["name"] for item in report["checks"] if not item["ok"]}
    assert "required_modality_image_text" in failed


def test_qwen_provider_metadata_matches_default_detector_shape() -> None:
    provider = Qwen25VLGuardrailProvider()

    assert provider.model_family == "qwen25_vl"
    assert provider.model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert provider.pooling == "text_tokens"
    assert provider.feature_dim == 2048


def test_llava_provider_metadata_matches_local_detector_shape() -> None:
    provider = LlavaOnevisionGuardrailProvider()

    assert provider.model_family == "llava_onevision"
    assert provider.model_id == "models\\huggingface\\llava-onevision-qwen2-0.5b-ov-hf"
    assert provider.pooling == "text_tokens"
    assert provider.feature_dim == 896


def test_target_validation_bundle_registry_records_status(tmp_path) -> None:
    gate_report = {"ok": True, "coverage": {"model_families": ["generic"]}}
    policy_path = tmp_path / "policy.json"
    provider_path = tmp_path / "provider.json"
    calibration_path = tmp_path / "calibration.json"
    monitoring_path = tmp_path / "monitoring.json"
    policy_path.write_text("{}", encoding="utf-8")
    provider_path.write_text('{"ok": true}', encoding="utf-8")
    calibration_path.write_text('{"feasible": true}', encoding="utf-8")
    monitoring_path.write_text('{"n_samples": 1}', encoding="utf-8")
    bundle = build_target_validation_bundle(
        target_name="unit-target",
        model_family="generic",
        model_id="unit-model",
        scope="controlled_deployment",
        detector_path="detector.npz",
        summary_paths=["summary.json"],
        gate_report=gate_report,
        policy_path=str(policy_path),
        provider_contract_path=str(provider_path),
        calibration_summary_path=str(calibration_path),
        monitoring_summary_path=str(monitoring_path),
        manifest_path="manifest.json",
    )

    registry = update_registry(tmp_path / "registry.json", bundle)
    markdown = render_target_bundle_markdown(bundle)

    assert bundle.status == "production_candidate"
    assert registry["status_counts"] == {"production_candidate": 1}
    assert "unit-target" in markdown
    assert "Intended modalities" in markdown


def test_target_release_validator_accepts_consistent_bundle(tmp_path) -> None:
    bundle_path = _write_release_fixture(tmp_path)

    report = validate_target_release(
        bundle_path,
        root=tmp_path,
        require_labeled_monitoring=True,
    )

    assert report["ok"] is True


def test_target_release_validator_rejects_policy_calibration_mismatch(tmp_path) -> None:
    bundle_path = _write_release_fixture(tmp_path)
    policy_path = tmp_path / "configs" / "policy.json"
    policy = _read_json(policy_path)
    policy["policy"]["block_threshold"] = 0.9
    policy_path.write_text(__import__("json").dumps(policy), encoding="utf-8")

    report = validate_target_release(bundle_path, root=tmp_path)

    assert report["ok"] is False
    failed = {item["name"] for item in report["checks"] if not item["ok"]}
    assert "policy_threshold_matches_calibration" in failed


def test_target_release_validator_enforces_provider_modality(tmp_path) -> None:
    bundle_path = _write_release_fixture(tmp_path)

    auto_passing = validate_target_release(bundle_path, root=tmp_path)
    passing = validate_target_release(
        bundle_path,
        root=tmp_path,
        required_provider_modalities=("text",),
    )
    failing = validate_target_release(
        bundle_path,
        root=tmp_path,
        required_provider_modalities=("image_text",),
    )

    assert auto_passing["ok"] is True
    assert passing["ok"] is True
    assert failing["ok"] is False
    failed = {item["name"] for item in failing["checks"] if not item["ok"]}
    assert "provider_contract_covers_modality:image_text" in failed


def test_target_release_validator_requires_supplemental_production_evidence(tmp_path) -> None:
    complete_bundle_path = _write_release_fixture(tmp_path / "complete")
    incomplete_bundle_path = _write_release_fixture(
        tmp_path / "incomplete",
        include_supplemental_evidence=False,
    )

    complete = validate_target_release(
        complete_bundle_path,
        root=tmp_path / "complete",
        require_robustness_evidence=True,
        require_attack_success_evidence=True,
    )
    incomplete = validate_target_release(
        incomplete_bundle_path,
        root=tmp_path / "incomplete",
        require_robustness_evidence=True,
        require_attack_success_evidence=True,
    )

    assert complete["ok"] is True
    assert incomplete["ok"] is False
    failed = {item["name"] for item in incomplete["checks"] if not item["ok"]}
    assert "robustness_evidence_present" in failed
    assert "attack_success_evidence_present" in failed


def test_target_registry_validator_accepts_release_audited_registry(tmp_path) -> None:
    bundle_path = _write_release_fixture(tmp_path)
    bundle = _read_json(tmp_path / bundle_path)
    _write_json(
        tmp_path / "docs" / "target_validation_registry.json",
        {
            "schema_version": 1,
            "status_counts": {"production_candidate": 1},
            "targets": [bundle],
        },
    )

    report = validate_target_registry(
        "docs/target_validation_registry.json",
        root=tmp_path,
        required_model_families=("generic",),
        required_target_modalities=("text",),
        require_labeled_monitoring=True,
        require_robustness_evidence=True,
        require_attack_success_evidence=True,
        required_provider_modalities=("text",),
    )

    assert report["ok"] is True
    assert report["registry"]["n_targets"] == 1
    assert report["coverage"]["target_modalities"] == ["text"]


def test_target_registry_validator_enforces_required_model_family(tmp_path) -> None:
    bundle_path = _write_release_fixture(tmp_path)
    bundle = _read_json(tmp_path / bundle_path)
    _write_json(
        tmp_path / "docs" / "target_validation_registry.json",
        {
            "schema_version": 1,
            "status_counts": {"production_candidate": 1},
            "targets": [bundle],
        },
    )

    report = validate_target_registry(
        "docs/target_validation_registry.json",
        root=tmp_path,
        required_model_families=("llava_onevision",),
    )

    assert report["ok"] is False
    failed = {item["name"] for item in report["checks"] if not item["ok"]}
    assert "required_model_families" in failed


def test_target_registry_validator_enforces_required_target_modality(tmp_path) -> None:
    bundle_path = _write_release_fixture(tmp_path)
    bundle = _read_json(tmp_path / bundle_path)
    _write_json(
        tmp_path / "docs" / "target_validation_registry.json",
        {
            "schema_version": 1,
            "status_counts": {"production_candidate": 1},
            "targets": [bundle],
        },
    )

    report = validate_target_registry(
        "docs/target_validation_registry.json",
        root=tmp_path,
        required_target_modalities=("image_text",),
    )

    assert report["ok"] is False
    failed = {item["name"] for item in report["checks"] if not item["ok"]}
    assert "required_target_modalities" in failed
    assert report["coverage"]["target_modalities"] == ["text"]


def test_deployment_package_round_trip_validates_release_artifacts(tmp_path) -> None:
    bundle_path = _write_release_fixture(tmp_path)
    package_dir = tmp_path / "deployment_package"

    build_report = build_target_deployment_package(
        bundle_path,
        package_dir,
        root=tmp_path,
        require_labeled_monitoring=True,
    )
    validate_report = validate_target_deployment_package(
        package_dir,
        require_labeled_monitoring=True,
        require_robustness_evidence=True,
        require_attack_success_evidence=True,
    )

    assert build_report["ok"] is True
    assert validate_report["ok"] is True
    assert (package_dir / "bundle.json").exists()
    assert (package_dir / "deployment_manifest.json").exists()
    assert validate_report["target"]["intended_modalities"] == ["text"]
    assert _read_json(package_dir / "bundle.json")["robustness_summary_paths"]
    assert _read_json(package_dir / "bundle.json")["attack_success_summary_paths"]


def _write_release_fixture(tmp_path, *, include_supplemental_evidence: bool = True) -> str:
    detector_path = tmp_path / "models" / "detector.npz"
    policy_path = tmp_path / "configs" / "policy.json"
    provider_path = tmp_path / "outputs" / "provider.json"
    calibration_path = tmp_path / "outputs" / "calibration.json"
    monitoring_path = tmp_path / "outputs" / "monitoring.json"
    summary_path = tmp_path / "outputs" / "summary.json"
    robustness_path = tmp_path / "outputs" / "robustness.json"
    attack_success_path = tmp_path / "outputs" / "attack_success.json"
    manifest_path = tmp_path / "docs" / "manifest.json"
    bundle_path = tmp_path / "docs" / "targets" / "unit-target.json"

    save_detector_artifact(detector_path, _artifact())
    _write_json(
        policy_path,
        {
            "schema_version": 1,
            "detector": "models/detector.npz",
            "policy": {
                "block_threshold": 0.5,
                "review_margin": 0.05,
                "action_on_error": "review",
                "require_matching_provenance": True,
            },
        },
    )
    _write_json(
        provider_path,
        {
            "ok": True,
            "provider": {
                "model_family": "generic",
                "model_id": "unit-model",
                "pooling": "text_tokens",
                "feature_dim": 2,
            },
            "detector": {
                "model_family": "generic",
                "model_id": "unit-model",
                "pooling": "text_tokens",
                "feature_dim": 2,
            },
            "request_coverage": {
                "n_requests": 1,
                "modalities": ["text"],
                "required_modalities": [],
                "requests": [
                    {
                        "request_id": "unit-request",
                        "modality": "text",
                        "has_text": True,
                        "n_images": 0,
                    }
                ],
            },
        },
    )
    _write_json(
        calibration_path,
        {
            "feasible": True,
            "n_samples": 4,
            "threshold": 0.5,
            "review_margin": 0.05,
            "metrics": {"auroc": 1.0, "auprc": 1.0, "precision": 1.0, "recall": 1.0},
        },
    )
    _write_json(
        monitoring_path,
        {
            "n_samples": 4,
            "action_counts": {"allow": 2, "block": 2},
            "labeled_metrics": {"auroc": 1.0},
        },
    )
    _write_json(
        summary_path,
        {
            "source": "unit",
            "n_samples": 4,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "metrics": {"auroc": 1.0, "auprc": 1.0, "precision": 1.0, "recall": 1.0},
            "detector": {
                "model_family": "generic",
                "model_id": "unit-model",
                "pooling": "text_tokens",
            },
        },
    )
    if include_supplemental_evidence:
        _write_json(
            robustness_path,
            {
                "source": "unit_robustness",
                "n_samples": 4,
                "metrics": {"auroc": 1.0, "auprc": 1.0},
            },
        )
        _write_json(
            attack_success_path,
            {
                "source": "unit_attack_success",
                "n_samples": 4,
                "absolute_asr_reduction": 1.0,
            },
        )
    manifest_artifacts = [
        {"path": "models/detector.npz"},
        {"path": "configs/policy.json"},
        {"path": "outputs/provider.json"},
        {"path": "outputs/calibration.json"},
        {"path": "outputs/monitoring.json"},
        {"path": "outputs/summary.json"},
        {"path": "docs/targets/unit-target.json"},
    ]
    if include_supplemental_evidence:
        manifest_artifacts.extend(
            [
                {"path": "outputs/robustness.json"},
                {"path": "outputs/attack_success.json"},
            ]
        )
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "artifacts": manifest_artifacts,
        },
    )
    bundle = {
        "target_name": "unit-target",
        "model_family": "generic",
        "model_id": "unit-model",
        "scope": "controlled_deployment",
        "status": "production_candidate",
        "intended_modalities": ["text"],
        "detector_path": "models/detector.npz",
        "policy_path": "configs/policy.json",
        "summary_paths": ["outputs/summary.json"],
        "provider_contract_path": "outputs/provider.json",
        "calibration_summary_path": "outputs/calibration.json",
        "monitoring_summary_path": "outputs/monitoring.json",
        "manifest_path": "docs/manifest.json",
        "gate_report": {
            "ok": True,
            "checks": [{"name": "unit", "ok": True, "detail": "ok"}],
        },
        "notes": "unit",
    }
    if include_supplemental_evidence:
        bundle["robustness_summary_paths"] = ["outputs/robustness.json"]
        bundle["attack_success_summary_paths"] = ["outputs/attack_success.json"]
    _write_json(bundle_path, bundle)
    return "docs/targets/unit-target.json"


def _write_json(path, payload) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))
