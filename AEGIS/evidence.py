from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EvidenceItem:
    key: str
    path: str
    kind: str
    exists: bool
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRequirement:
    key: str
    description: str
    required_evidence_keys: tuple[str, ...]
    status: str
    evidence_paths: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EVIDENCE_SPECS = {
    "default_detector": ("models/aegis/aegis_qwen25vl3b_text_detector.npz", "artifact"),
    "llava_detector": ("models/aegis/aegis_llava_onevision_05b_text_detector.npz", "artifact"),
    "reproducibility_manifest": ("docs/reproducibility_manifest.json", "manifest"),
    "expanded_litmus_summary": (
        "outputs/aegis_classify_multimodal_litmus_v1_summary.json",
        "summary_json",
    ),
    "expanded_litmus_metadata": (
        "data/processed/aegis_classify_multimodal_litmus_v1/metadata.csv",
        "metadata_csv",
    ),
    "vlguard_metadata": ("data/processed/vlguard_test_full_groupsplit_metadata.csv", "metadata_csv"),
    "mssbench_metadata": ("data/processed/mssbench_full_groupsplit_metadata.csv", "metadata_csv"),
    "jailbreakv_metadata": (
        "data/processed/jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv",
        "metadata_csv",
    ),
    "qwen_low_label": (
        "outputs/vlguard_test_full_qwen25vl3b_low_label_seed_sweep_summary.csv",
        "result_csv",
    ),
    "trusted_pseudo_low_label": (
        "outputs/mssbench_full_qwen25vl3b_trusted_pseudo_logistic_seed_sweep_summary.csv",
        "result_csv",
    ),
    "prompt_robustness": (
        "outputs/jailbreakv_eval_image_matched_prompt_robustness_evaluation.csv",
        "result_csv",
    ),
    "adaptive_prompt": (
        "outputs/jailbreakv_eval_image_matched_adaptive_prompt_evaluation.csv",
        "result_csv",
    ),
    "guardrail_asr": (
        "outputs/jailbreakv_eval_image_matched_llava_onevision_05b_guardrail_asr_summary.csv",
        "result_csv",
    ),
    "qwen_signal_analysis": (
        "outputs/jailbreakv_eval_image_matched_qwen25vl3b_signal_analysis_feature_comparison.csv",
        "result_csv",
    ),
    "mssbench_signal_analysis": (
        "outputs/mssbench_full_qwen25vl3b_signal_analysis_feature_comparison.csv",
        "result_csv",
    ),
    "llava_signal_analysis": (
        "outputs/jailbreakv_eval_image_matched_llava_onevision_05b_signal_analysis_feature_comparison.csv",
        "result_csv",
    ),
    "qwen_transfer": (
        "outputs/jailbreakv_source_disjoint_transfer_summary.csv",
        "result_csv",
    ),
    "llava_attack_holdout": (
        "outputs/jailbreakv_eval_720_image_matched_controls_llava_onevision_05b_attack_family_holdout_summary.csv",
        "result_csv",
    ),
    "production_playbook": ("docs/production_guardrail_playbook.md", "doc"),
    "policy_config": ("configs/guardrail_policy.example.json", "config"),
    "guardrail_runtime": ("AEGIS/guardrail.py", "script"),
    "provider_contract_runtime": ("AEGIS/provider_contract.py", "script"),
    "deployment_package_runtime": ("AEGIS/deployment_package.py", "script"),
    "target_release_runtime": ("AEGIS/target_release.py", "script"),
    "provider_guardrail_server": ("scripts/serve_guardrail_provider.py", "script"),
    "detector_feature_evaluator": ("scripts/evaluate_detector_features.py", "script"),
    "target_release_validator": ("scripts/validate_target_release.py", "script"),
    "target_registry_validator": ("scripts/validate_target_registry.py", "script"),
    "target_deployment_package_builder": (
        "scripts/build_target_deployment_package.py",
        "script",
    ),
    "target_deployment_package_validator": (
        "scripts/validate_target_deployment_package.py",
        "script",
    ),
    "qwen_deployment_package_report": (
        "outputs/deployment_packages/qwen25vl3b_controlled/deployment_package_report.json",
        "summary_json",
    ),
    "llava_deployment_package_report": (
        "outputs/deployment_packages/llava_onevision_05b_controlled/deployment_package_report.json",
        "summary_json",
    ),
    "llava_policy": ("configs/llava_onevision_05b_controlled_policy.json", "config"),
    "llava_provider_contract": ("outputs/llava_onevision_05b_provider_contract.json", "summary_json"),
    "llava_calibration_summary": (
        "outputs/llava_onevision_05b_controlled_calibration_summary.json",
        "summary_json",
    ),
    "llava_monitoring_summary": (
        "outputs/llava_onevision_05b_controlled_monitoring_summary.json",
        "summary_json",
    ),
    "llava_target_bundle": ("docs/targets/llava_onevision_05b_controlled.json", "manifest"),
    "llava_controlled_summary": ("outputs/llava_onevision_05b_controlled_summary.json", "summary_json"),
    "qwen_policy": ("configs/qwen25vl3b_controlled_policy.json", "config"),
    "qwen_provider_contract": ("outputs/qwen25vl3b_provider_contract.json", "summary_json"),
    "qwen_calibration_summary": (
        "outputs/qwen25vl3b_controlled_calibration_summary.json",
        "summary_json",
    ),
    "qwen_monitoring_summary": (
        "outputs/qwen25vl3b_controlled_monitoring_summary.json",
        "summary_json",
    ),
    "qwen_target_bundle": ("docs/targets/qwen25vl3b_controlled.json", "manifest"),
    "target_validation_registry": ("docs/target_validation_registry.json", "manifest"),
    "target_portfolio_universal_report": (
        "outputs/target_registry_universal_portfolio_report.json",
        "summary_json",
    ),
}


REQUIREMENT_SPECS = [
    (
        "balanced_corpus",
        "Balanced benign/adversarial corpus across VLGuard, MSSBench, JailBreakV, and litmus data.",
        ("vlguard_metadata", "mssbench_metadata", "jailbreakv_metadata", "expanded_litmus_metadata"),
        "Dataset metadata exists locally; third-party raw data remains untracked by design.",
    ),
    (
        "red_teaming_attack_styles",
        "Red-team prompts cover obfuscation, role-play, image-embedded cues, privacy, harassment, and harmful instructions.",
        ("expanded_litmus_metadata", "prompt_robustness", "adaptive_prompt"),
        "Expanded litmus and robustness/adaptive variants provide safe redacted coverage.",
    ),
    (
        "representation_detector",
        "Representation-space detector artifact and reproducible scoring pipeline.",
        ("default_detector", "reproducibility_manifest"),
        "Detector artifact and manifest are present.",
    ),
    (
        "uncertainty_and_review",
        "Guardrail decisions include uncertainty/review routing and uncertainty evidence.",
        ("expanded_litmus_summary", "qwen_signal_analysis"),
        "Runtime supports review decisions; litmus summary reports uncertain samples.",
    ),
    (
        "signal_importance",
        "Signal importance, cross-modal consistency, and modality attribution are evaluated.",
        ("qwen_signal_analysis", "mssbench_signal_analysis", "llava_signal_analysis"),
        "Signal-analysis result files exist across jailbreak and MSSBench settings.",
    ),
    (
        "low_label_learning",
        "Low-label and pseudo-label/self-training variants reduce annotation cost.",
        ("qwen_low_label", "trusted_pseudo_low_label"),
        "Low-label and trusted-plus-pseudo summaries exist.",
    ),
    (
        "diverse_models",
        "Evaluation covers more than one MLLM representation family.",
        ("expanded_litmus_summary", "llava_controlled_summary", "llava_attack_holdout", "guardrail_asr"),
        "Qwen2.5-VL and LLaVA-OneVision controlled/release evidence are present.",
    ),
    (
        "diverse_datasets",
        "Evaluation covers multiple datasets and transfer settings.",
        ("vlguard_metadata", "mssbench_metadata", "jailbreakv_metadata", "qwen_transfer"),
        "VLGuard, MSSBench, JailBreakV, and transfer summaries are present.",
    ),
    (
        "robustness",
        "Robustness is tested under deterministic and bounded adaptive prompt attacks.",
        ("prompt_robustness", "adaptive_prompt"),
        "Prompt robustness and bounded adaptive evaluation files exist.",
    ),
    (
        "attack_success_reduction",
        "Guardrail blocking reduces response-level attack-success proxy.",
        ("guardrail_asr",),
        "ASR proxy summary exists for LLaVA-OneVision guarded response evaluation.",
    ),
    (
        "production_operations",
        "Deployment path includes policy config, calibration, monitoring, and release gating.",
        (
            "production_playbook",
            "policy_config",
            "guardrail_runtime",
            "provider_contract_runtime",
            "deployment_package_runtime",
            "target_release_runtime",
            "provider_guardrail_server",
            "detector_feature_evaluator",
            "target_release_validator",
            "target_registry_validator",
            "target_deployment_package_builder",
            "target_deployment_package_validator",
            "qwen_deployment_package_report",
            "llava_deployment_package_report",
            "llava_policy",
            "llava_provider_contract",
            "llava_calibration_summary",
            "llava_monitoring_summary",
            "llava_target_bundle",
            "qwen_policy",
            "qwen_provider_contract",
            "qwen_calibration_summary",
            "qwen_monitoring_summary",
            "qwen_target_bundle",
            "reproducibility_manifest",
            "target_validation_registry",
        ),
        "Operational docs/config, provider contracts, calibrated policies, monitoring baselines, modality-declared target bundles, release validators, deployment packages, and manifest exist.",
    ),
    (
        "universal_any_llm_mllm_claim",
        "Validated on any target LLM/MLLM through per-model embedding providers and target traffic calibration.",
        (
            "production_playbook",
            "qwen_target_bundle",
            "llava_target_bundle",
            "target_validation_registry",
            "target_portfolio_universal_report",
        ),
        "The registry portfolio audit is executable and currently records image-text production coverage; this remains partial because a real text-only LLM target, Qwen target-specific attack-success evidence, and arbitrary additional target models still need their own provider contract, calibrated policy, representative traffic evidence, robustness evidence, and guarded attack-success evidence.",
    ),
]


def build_evidence_report(root: str | Path = ".") -> dict[str, object]:
    root_path = Path(root)
    evidence = {
        key: _describe_evidence(key, root_path / path, kind)
        for key, (path, kind) in EVIDENCE_SPECS.items()
    }
    requirements = [
        _evaluate_requirement(key, description, required, notes, evidence)
        for key, description, required, notes in REQUIREMENT_SPECS
    ]
    status_counts: dict[str, int] = {}
    for requirement in requirements:
        status_counts[requirement.status] = status_counts.get(requirement.status, 0) + 1

    return {
        "schema_version": 1,
        "overall_status": _overall_status(requirements),
        "status_counts": status_counts,
        "evidence": {key: item.to_dict() for key, item in evidence.items()},
        "requirements": [requirement.to_dict() for requirement in requirements],
    }


def render_markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# AEGIS Production Evidence Report",
        "",
        f"Overall status: `{report['overall_status']}`",
        "",
        "## Requirement Matrix",
        "",
        "| Requirement | Status | Evidence | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for requirement in report["requirements"]:
        evidence_paths = "<br>".join(requirement["evidence_paths"]) or "None"
        lines.append(
            "| {key} | `{status}` | {evidence} | {notes} |".format(
                key=requirement["key"],
                status=requirement["status"],
                evidence=evidence_paths,
                notes=requirement["notes"],
            )
        )

    lines.extend(["", "## Evidence Inventory", ""])
    for key, item in report["evidence"].items():
        exists = "yes" if item["exists"] else "no"
        lines.append(f"- `{key}`: {exists} - `{item['path']}`")
    lines.append("")
    return "\n".join(lines)


def _describe_evidence(key: str, path: Path, kind: str) -> EvidenceItem:
    exists = path.exists()
    summary: dict[str, object] = {}
    if exists:
        try:
            if kind in {"metadata_csv", "result_csv"}:
                summary = _summarize_csv(path)
            elif kind in {"summary_json", "manifest", "config"}:
                summary = _summarize_json(path)
            else:
                summary = {"bytes": int(path.stat().st_size)}
        except Exception as exc:
            summary = {"error": str(exc)}
    return EvidenceItem(
        key=key,
        path=str(path),
        kind=kind,
        exists=exists,
        summary=summary,
    )


def _evaluate_requirement(
    key: str,
    description: str,
    required_evidence_keys: tuple[str, ...],
    notes: str,
    evidence: dict[str, EvidenceItem],
) -> EvidenceRequirement:
    present = [evidence[item] for item in required_evidence_keys if evidence[item].exists]
    if len(present) == len(required_evidence_keys):
        status = "satisfied"
    elif present:
        status = "partial"
    else:
        status = "missing"
    if key == "universal_any_llm_mllm_claim":
        status = "partial"
    return EvidenceRequirement(
        key=key,
        description=description,
        required_evidence_keys=required_evidence_keys,
        status=status,
        evidence_paths=tuple(item.path for item in present),
        notes=notes,
    )


def _overall_status(requirements: list[EvidenceRequirement]) -> str:
    if any(requirement.status == "missing" for requirement in requirements):
        return "incomplete"
    if any(requirement.status == "partial" for requirement in requirements):
        return "partial"
    return "satisfied"


def _summarize_csv(path: Path) -> dict[str, object]:
    table = pd.read_csv(path)
    summary: dict[str, object] = {
        "rows": int(len(table)),
        "columns": list(table.columns),
    }
    for column in ("label", "source", "attack_style", "harm_category", "modality"):
        if column in table.columns:
            counts = table[column].astype(str).value_counts(dropna=False).sort_index()
            summary[f"{column}_counts"] = {
                str(key): int(value) for key, value in counts.items()
            }
    return summary


def _summarize_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    summary_keys = [
        "schema_version",
        "source",
        "n_samples",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "metrics",
        "detector",
        "project",
        "git",
        "ok",
        "feasible",
        "threshold",
        "status",
        "status_counts",
        "target_name",
        "target",
        "package_dir",
        "provider",
        "coverage",
        "calibration",
        "action_counts",
        "verdict_counts",
        "labeled_metrics",
    ]
    return {key: payload[key] for key in summary_keys if key in payload}
