from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.detector_artifact import load_detector_artifact
from AEGIS.readiness import inspect_detector_artifact, sha256_file


DEFAULT_OUTPUT = "docs/reproducibility_manifest.json"

DEFAULT_PATHS = [
    "models/aegis/aegis_qwen25vl3b_text_detector.npz",
    "models/aegis/aegis_llava_onevision_05b_text_detector.npz",
    "configs/guardrail_policy.example.json",
    "configs/qwen25vl3b_controlled_policy.json",
    "configs/llava_onevision_05b_controlled_policy.json",
    "AEGIS/guardrail.py",
    "AEGIS/provider_contract.py",
    "AEGIS/providers.py",
    "AEGIS/service.py",
    "AEGIS/deployment_package.py",
    "AEGIS/target_release.py",
    "AEGIS/target_validation.py",
    "docs/production_guardrail_playbook.md",
    "docs/target_validation_registry.json",
    "docs/targets/llava_onevision_05b_controlled.json",
    "docs/targets/llava_onevision_05b_controlled.md",
    "docs/targets/qwen25vl3b_controlled.json",
    "docs/targets/qwen25vl3b_controlled.md",
    "scripts/build_production_evidence_report.py",
    "scripts/build_target_validation_bundle.py",
    "scripts/calibrate_guardrail_policy.py",
    "scripts/evaluate_detector_features.py",
    "scripts/monitor_guardrail_decisions.py",
    "scripts/serve_guardrail_features.py",
    "scripts/serve_guardrail_provider.py",
    "scripts/validate_production_readiness.py",
    "scripts/validate_embedding_provider_contract.py",
    "scripts/validate_target_release.py",
    "scripts/validate_target_registry.py",
    "scripts/build_target_deployment_package.py",
    "scripts/validate_target_deployment_package.py",
    "outputs/deployment_packages/qwen25vl3b_controlled/bundle.json",
    "outputs/deployment_packages/qwen25vl3b_controlled/deployment_manifest.json",
    "outputs/deployment_packages/qwen25vl3b_controlled/deployment_package_report.json",
    "outputs/deployment_packages/llava_onevision_05b_controlled/bundle.json",
    "outputs/deployment_packages/llava_onevision_05b_controlled/deployment_manifest.json",
    "outputs/deployment_packages/llava_onevision_05b_controlled/deployment_package_report.json",
    "outputs/target_registry_universal_portfolio_report.json",
    "data/processed/aegis_inference_calibration_v1_metadata.csv",
    "data/processed/aegis_classify_multimodal_litmus_v1/metadata.csv",
    "outputs/aegis_classify_multimodal_litmus_v1_summary.json",
    "outputs/aegis_classify_multimodal_litmus_v1_scores.csv",
    "outputs/aegis_inference_calibration_v1_qwen25vl3b_text_tokens.npz",
    "outputs/llava_onevision_05b_controlled_summary.json",
    "outputs/llava_onevision_05b_controlled_scores.csv",
    "outputs/llava_onevision_05b_provider_contract.json",
    "outputs/llava_onevision_05b_controlled_calibration_summary.json",
    "outputs/llava_onevision_05b_controlled_monitoring_summary.json",
    "outputs/jailbreakv_eval_720_image_matched_controls_llava_onevision_05b/jailbreakv_eval_720_image_matched_controls_llava_onevision_05b_layer_m1_text_tokens.npz",
    "outputs/qwen25vl3b_provider_contract.json",
    "outputs/qwen25vl3b_controlled_calibration_summary.json",
    "outputs/qwen25vl3b_controlled_decisions.csv",
    "outputs/qwen25vl3b_controlled_decision_summary.json",
    "outputs/qwen25vl3b_controlled_monitoring_summary.json",
    "data/processed/vlguard_test_full_groupsplit_metadata.csv",
    "data/processed/mssbench_full_groupsplit_metadata.csv",
    "data/processed/jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv",
    "outputs/vlguard_test_full_qwen25vl3b_low_label_seed_sweep_summary.csv",
    "outputs/mssbench_full_qwen25vl3b_trusted_pseudo_logistic_seed_sweep_summary.csv",
    "outputs/jailbreakv_eval_image_matched_prompt_robustness_evaluation.csv",
    "outputs/jailbreakv_eval_image_matched_adaptive_prompt_evaluation.csv",
    "outputs/jailbreakv_eval_720_image_matched_controls_llava_onevision_05b_attack_family_holdout_summary.csv",
    "outputs/jailbreakv_eval_image_matched_llava_onevision_05b_guardrail_asr_summary.csv",
]

COMMAND_RECIPES = [
    {
        "name": "build_expanded_litmus_panel",
        "command": (
            "python scripts/build_classify_litmus_panel.py "
            "--output-dir data/processed/aegis_classify_multimodal_litmus_v1"
        ),
        "notes": "Generates the balanced image-text litmus metadata and card images.",
    },
    {
        "name": "run_litmus_evaluation",
        "command": (
            "python scripts/run_classify_litmus.py "
            "--refresh-embeddings --require-perfect"
        ),
        "notes": "Requires local MLLM weights/cache and writes scores plus summary JSON.",
    },
    {
        "name": "train_default_detector",
        "command": (
            "aegis train-detector --threshold-strategy max-fpr --max-fpr 0.01 "
            "--uncertainty-margin 0.05 --source "
            "vlguard_mssbench_jailbreakv_generic_text_qwen25vl3b"
        ),
        "notes": "See README for full embedding and metadata path lists.",
    },
    {
        "name": "local_quality_gate",
        "command": (
            "python -m ruff check AEGIS scripts tests && "
            "python -m pytest && "
            "python scripts/run_smoke_tests.py"
        ),
        "notes": "Fast CPU-only quality gate; does not download models or datasets.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a raw-prompt-free reproducibility manifest for local AEGIS artifacts."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        default=None,
        help="Additional artifact path to describe. Can be passed more than once.",
    )
    args = parser.parse_args()

    paths = [Path(path) for path in DEFAULT_PATHS]
    if args.paths:
        paths.extend(Path(path) for path in args.paths)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "project": _project_metadata(),
        "git": _git_metadata(),
        "detector": _detector_metadata(Path("models/aegis/aegis_qwen25vl3b_text_detector.npz")),
        "artifacts": [_describe_path(path) for path in paths],
        "command_recipes": COMMAND_RECIPES,
        "safety_note": (
            "CSV descriptions include counts, hashes, and provenance only. Raw prompt text "
            "and generated model responses are intentionally omitted."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "n_artifacts": len(paths)}, sort_keys=True))


def _project_metadata() -> dict[str, Any]:
    project: dict[str, str] = {}
    in_project = False
    for line in Path("pyproject.toml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = False
            continue
        if not in_project or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        project[key.strip()] = value.strip().strip('"')
    return {
        "name": project.get("name"),
        "version": project.get("version"),
        "requires_python": project.get("requires-python"),
    }


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _run_git("rev-parse", "HEAD"),
        "branch": _run_git("branch", "--show-current"),
        "dirty": bool(_run_git("status", "--short")),
    }


def _run_git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _detector_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    info = inspect_detector_artifact(path)
    artifact = load_detector_artifact(path)
    return {
        **info,
        "classifier": {
            "standardize": bool(artifact.classifier.standardize),
            "random_seed": int(artifact.classifier.random_seed),
        },
    }


def _describe_path(path: Path) -> dict[str, Any]:
    description: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return description
    description.update(
        {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "modified_utc": datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc,
            )
            .replace(microsecond=0)
            .isoformat(),
        }
    )
    if path.suffix.lower() == ".csv":
        description["csv"] = _describe_csv(path)
    elif path.suffix.lower() == ".json":
        description["json"] = _describe_json(path)
    elif path.suffix.lower() == ".npz":
        description["npz"] = _describe_npz(path)
    return description


def _describe_csv(path: Path) -> dict[str, Any]:
    table = pd.read_csv(path)
    summary: dict[str, Any] = {
        "rows": int(len(table)),
        "columns": list(table.columns),
    }
    for column in [
        "label",
        "experiment_split",
        "source",
        "harm_category",
        "attack_style",
        "modality",
    ]:
        if column in table.columns:
            counts = table[column].astype(str).value_counts(dropna=False).sort_index()
            summary[f"{column}_counts"] = {
                str(key): int(value) for key, value in counts.items()
            }
    return summary


def _describe_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    safe_keys = [
        "source",
        "panel_version",
        "n_samples",
        "n_benign",
        "n_malicious",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "perfect",
        "metrics",
        "detector",
    ]
    return {
        "keys": sorted(payload),
        "summary": {key: payload[key] for key in safe_keys if key in payload},
    }


def _describe_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            key: {
                "shape": list(np.asarray(data[key]).shape),
                "dtype": str(np.asarray(data[key]).dtype),
            }
            for key in sorted(data.files)
        }
        provenance = {
            key: _scalar_string(data, key)
            for key in ["model_id", "layer", "pooling", "source", "artifact_version"]
            if key in data
        }
    return {"arrays": arrays, "provenance": provenance}


def _scalar_string(data, key: str) -> str:
    values = np.asarray(data[key]).reshape(-1)
    if len(values) == 0:
        return ""
    return str(values[0])


if __name__ == "__main__":
    main()
