from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree

import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
MANIFEST_PATH = ROOT / "MANIFEST.json"
REPORT_PATH = ROOT / "verification_report.json"
MANIFEST_SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 3
VERIFIER_VERSION = "3.0.0"
FLOAT_TOLERANCE = 1e-12
RUNTIME_ARTIFACT_SHA256 = (
    "9816ed0b706b3d0e01162701310ad77bc795c7dd99f7e2e6da4133d6fa299587"
)

REQUIRED_DIRECTORIES = {
    "ablation_report",
    "benchmark",
    "ci",
    "pipeline",
    "red_teaming",
    "runtime_validation",
}
ALLOWED_TOP_LEVEL_FILES = {
    "MANIFEST.json",
    "README.md",
    "verification_report.json",
    "verify_deliverables.py",
}
EXCLUDED_DIRECTORY_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "reproductions",
    "runs",
}
EXCLUDED_FILE_NAMES = {
    ".DS_Store",
    "MANIFEST.json",
    "verification_report.json",
}
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".mjs",
    ".ndjson",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

PIPELINE_FILES = {
    "README.md",
    "aegis_research/__init__.py",
    "aegis_research/experiment.py",
    "aegis_research/io.py",
    "aegis_research/metrics.py",
    "aegis_research/model.py",
    "aegis_research/signals.py",
    "configs/default.json",
    "pyproject.toml",
    "requirements.txt",
    "schemas/feature_bundle.schema.json",
    "scripts/extract_mllm_features.py",
    "scripts/run_experiment.py",
    "scripts/score_feature_bundle.py",
    "tests/test_pipeline.py",
}
REDTEAM_FILES = {
    "README.md",
    "SAFE_USE.md",
    "attack_catalog.json",
    "evaluate_adaptive.py",
    "evaluate_attack_success.py",
    "generate_variants.py",
    "response_judgment_schema.json",
}
ABLATION_FILES = {
    "README.md",
    "run_ablation.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the current AEGIS Final Report deliverables."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Read-only verification (the default).",
    )
    mode.add_argument(
        "--update",
        action="store_true",
        help="Refresh MANIFEST.json and verification_report.json after reviewed changes.",
    )
    args = parser.parse_args()
    update = bool(args.update)

    checks: list[dict[str, object]] = []
    suites: tuple[tuple[str, Callable[[list[dict[str, object]]], None]], ...] = (
        ("scope", _verify_scope),
        ("benchmark", _verify_benchmark),
        ("pipeline", _verify_pipeline),
        ("redteam", _verify_redteam),
        ("ablation", _verify_ablation),
        ("runtime", _verify_runtime),
        ("source", _verify_source_files),
    )
    for suite_name, suite in suites:
        try:
            suite(checks)
        except Exception as exc:
            _record(
                checks,
                f"{suite_name}_suite_completed",
                False,
                f"{type(exc).__name__}: {exc}",
            )

    manifest = _build_manifest()
    if update:
        if not _all_passed(checks):
            _print_result("update", checks, manifest)
            raise SystemExit(1)
        _write_json(MANIFEST_PATH, manifest)
        _record(
            checks,
            "manifest_update_round_trip",
            _load_json(MANIFEST_PATH) == manifest,
            manifest["artifact_set_sha256"],
        )
        report = _build_stored_report(checks, manifest)
        _write_json(REPORT_PATH, report)
        _print_result("update", checks, manifest, report)
    else:
        _verify_stored_manifest(checks, manifest)
        _verify_stored_report(checks, manifest)
        _print_result("check", checks, manifest)

    if not _all_passed(checks):
        raise SystemExit(1)


def _verify_scope(checks: list[dict[str, object]]) -> None:
    directories = {path.name for path in ROOT.iterdir() if path.is_dir()}
    files = {path.name for path in ROOT.iterdir() if path.is_file()}
    unexpected_directories = sorted(
        directories - REQUIRED_DIRECTORIES - {"reproductions", "runs"}
    )
    unexpected_files = sorted(files - ALLOWED_TOP_LEVEL_FILES)
    _record(
        checks,
        "scope_required_directories",
        REQUIRED_DIRECTORIES <= directories,
        sorted(directories),
    )
    _record(
        checks,
        "scope_no_unexpected_top_level_entries",
        not unexpected_directories and not unexpected_files,
        {
            "directories": unexpected_directories,
            "files": unexpected_files,
        },
    )

    links: list[str] = []
    for directory, names, filenames in os.walk(ROOT, followlinks=False):
        for name in [*names, *filenames]:
            path = Path(directory) / name
            is_junction = getattr(path, "is_junction", lambda: False)()
            if path.is_symlink() or is_junction:
                links.append(path.relative_to(ROOT).as_posix())
    _record(checks, "scope_no_symlinks_or_junctions", not links, links)

    generated = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if (
            path.name in {"__pycache__", ".pytest_cache", "node_modules"}
            or path.suffix in {".pyc", ".pyo"}
            or path.name.endswith(".xlsx.inspect.ndjson")
        )
    ]
    _record(checks, "scope_no_generated_scratch", not generated, generated)

    attributes = (REPOSITORY / ".gitattributes").read_text(encoding="utf-8")
    required_patterns = ("*.npz binary", "*.png binary", "*.xlsx binary")
    _record(
        checks,
        "scope_binary_attributes",
        all(pattern in attributes for pattern in required_patterns),
    )


def _verify_benchmark(checks: list[dict[str, object]]) -> None:
    root = ROOT / "benchmark"
    csv_path = root / "data" / "benchmark.csv"
    json_path = root / "data" / "benchmark_rows.json"
    rows = _read_csv(csv_path)
    json_rows = _load_json(json_path)
    schema = _load_json(root / "schema.json")
    header = list(rows[0]) if rows else []

    _record(checks, "benchmark_row_count", len(rows) == 64, len(rows))
    _record(
        checks,
        "benchmark_schema_columns",
        set(schema["required"]) == set(header),
        {
            "missing": sorted(set(schema["required"]) - set(header)),
            "extra": sorted(set(header) - set(schema["required"])),
        },
    )
    _record(
        checks,
        "benchmark_schema_relationship_rules",
        isinstance(schema.get("allOf"), list) and len(schema["allOf"]) >= 2,
    )

    csv_equivalent = [
        {key: _csv_text(row.get(key)) for key in header}
        for row in json_rows
    ]
    _record(
        checks,
        "benchmark_csv_json_full_alignment",
        rows == csv_equivalent,
    )
    _record(
        checks,
        "benchmark_balanced_labels",
        Counter(row["label"] for row in rows)
        == Counter({"benign": 32, "malicious": 32}),
    )
    expected_splits = Counter(
        {
            ("train", "benign"): 16,
            ("train", "malicious"): 16,
            ("validation", "benign"): 8,
            ("validation", "malicious"): 8,
            ("test", "benign"): 8,
            ("test", "malicious"): 8,
        }
    )
    _record(
        checks,
        "benchmark_split_label_balance",
        Counter((row["split"], row["label"]) for row in rows) == expected_splits,
    )
    _record(
        checks,
        "benchmark_unique_identifiers",
        len({row["sample_id"] for row in rows}) == len(rows),
    )

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    groups_ok = len(groups) == 32 and all(
        len(group) == 2
        and {row["label_id"] for row in group} == {"0", "1"}
        and len({row["split"] for row in group}) == 1
        for group in groups.values()
    )
    _record(checks, "benchmark_pair_split_isolation", groups_ok, len(groups))

    attack_counts = Counter(
        row["attack_style"] for row in rows if row["label_id"] == "1"
    )
    _record(
        checks,
        "benchmark_attack_style_coverage",
        len(attack_counts) == 8 and set(attack_counts.values()) == {4},
        dict(sorted(attack_counts.items())),
    )
    _record(
        checks,
        "benchmark_modality_counts",
        Counter(row["modality"] for row in rows)
        == Counter({"image_text": 44, "text": 20}),
    )

    content_errors: list[str] = []
    image_count = 0
    for row in rows:
        normalized = " ".join(row["prompt_text"].lower().split())
        if hashlib.sha256(normalized.encode("utf-8")).hexdigest() != row[
            "prompt_sha256"
        ]:
            content_errors.append(f"{row['sample_id']}:prompt_hash")
        if row["label"] == "benign":
            if row["label_id"] != "0" or row["harm_category"] != "none":
                content_errors.append(f"{row['sample_id']}:label_relation")
        elif row["label_id"] != "1":
            content_errors.append(f"{row['sample_id']}:label_relation")
        if row["modality"] == "text":
            if row["image_path"] or row["image_sha256"]:
                content_errors.append(f"{row['sample_id']}:text_image_fields")
        else:
            image_count += 1
            image_path = root / row["image_path"]
            if (
                not image_path.is_file()
                or _raw_sha256(image_path) != row["image_sha256"]
            ):
                content_errors.append(f"{row['sample_id']}:image_hash")
    _record(
        checks,
        "benchmark_relationships_and_hashes",
        not content_errors and image_count == 44,
        content_errors or {"images": image_count},
    )

    asset_manifest = _load_json(root / "data" / "asset_manifest.json")
    asset_manifest_ok = (
        asset_manifest["rows"] == 64
        and asset_manifest["image_count"] == 44
        and asset_manifest["label_counts"] == {"benign": 32, "malicious": 32}
        and asset_manifest["rows_json_sha256"] == _raw_sha256(json_path)
        and asset_manifest["csv_sha256"] == _raw_sha256(csv_path)
    )
    _record(checks, "benchmark_asset_manifest", asset_manifest_ok)

    xlsx_path = root / "benchmark.xlsx"
    formula_count = 0
    xlsx_errors: list[str] = []
    sheet_names: list[str] = []
    with zipfile.ZipFile(xlsx_path) as archive:
        names = set(archive.namelist())
        workbook_root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheet_names = [
            element.attrib["name"]
            for element in workbook_root.iter()
            if element.tag.endswith("}sheet")
        ]
        for name in sorted(
            item
            for item in names
            if item.startswith("xl/worksheets/") and item.endswith(".xml")
        ):
            payload = archive.read(name)
            root_element = ElementTree.fromstring(payload)
            formula_count += sum(
                1 for element in root_element.iter() if element.tag.endswith("}f")
            )
            text = payload.decode("utf-8", errors="replace")
            for token in ("#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"):
                if token in text:
                    xlsx_errors.append(f"{name}:{token}")
    _record(
        checks,
        "benchmark_workbook_structure",
        sheet_names == ["Summary", "Benchmark", "Data Dictionary"]
        and formula_count >= 20
        and not xlsx_errors,
        {
            "sheets": sheet_names,
            "formulas": formula_count,
            "errors": xlsx_errors,
        },
    )

    qa_files = (
        root / "qa" / "summary.png",
        root / "qa" / "benchmark_head.png",
        root / "qa" / "benchmark_tail.png",
        root / "qa" / "data_dictionary.png",
    )
    qa_text = (root / "qa" / "workbook_inspect.ndjson").read_text(
        encoding="utf-8"
    )
    _record(
        checks,
        "benchmark_workbook_qa",
        all(path.stat().st_size > 10_000 for path in qa_files)
        and '"Balanced?","PASS"' in qa_text
        and "Evidence boundary" in qa_text
        and "Cell search matched 0 entries" in qa_text,
    )


def _verify_pipeline(checks: list[dict[str, object]]) -> None:
    root = ROOT / "pipeline"
    actual_files = _relative_file_set(root)
    _record(
        checks,
        "pipeline_current_tooling_layout",
        actual_files == PIPELINE_FILES,
        {
            "missing": sorted(PIPELINE_FILES - actual_files),
            "unexpected": sorted(actual_files - PIPELINE_FILES),
        },
    )

    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    _record(
        checks,
        "pipeline_dependencies_pinned",
        requirements.splitlines() == ["numpy==2.3.5", "Pillow==12.2.0"],
    )

    schema = _load_json(root / "schemas" / "feature_bundle.schema.json")
    required_arrays = {
        "sample_ids",
        "text_embeddings",
        "image_embeddings",
        "attribution_features",
        "feature_source",
    }
    properties = schema.get("properties", {})
    provenance_fields = {
        "model_family",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "layer",
        "pooling",
        "preprocessing_sha256",
        "id_column",
    }
    schema_ok = (
        required_arrays | provenance_fields <= set(schema.get("required", []))
        and required_arrays | provenance_fields <= set(properties)
        and all(
            properties[name].get("type") == "array"
            for name in (
                "sample_ids",
                "text_embeddings",
                "image_embeddings",
                "attribution_features",
            )
        )
        and properties["preprocessing_sha256"].get("pattern")
        == "^[a-f0-9]{64}$"
        and set(properties["id_column"].get("enum", []))
        == {"sample_id", "variant_id"}
    )
    _record(checks, "pipeline_feature_bundle_schema", schema_ok)

    config = _load_json(root / "configs" / "default.json")
    config_ok = (
        config["seed"] == 42
        and config["threshold_selection"] == "validation_max_f1_then_precision"
        and 0.0 < float(config["learning_rate"]) < 1.0
        and int(config["epochs"]) > 0
        and 0.0 <= float(config["pseudo_low"]) < float(config["pseudo_high"]) <= 1.0
    )
    _record(checks, "pipeline_default_experiment_config", config_ok)

    extraction = (root / "scripts" / "extract_mllm_features.py").read_text(
        encoding="utf-8"
    )
    experiment = (root / "scripts" / "run_experiment.py").read_text(
        encoding="utf-8"
    )
    scoring = (root / "scripts" / "score_feature_bundle.py").read_text(
        encoding="utf-8"
    )
    entrypoints_ok = (
        all(
            token in extraction
            for token in (
                "--model-family",
                "--model-revision",
                "--tokenizer-revision",
                "--output-dir",
                "preprocessing_sha256",
                "variant_id",
            )
        )
        and all(
            token in experiment
            for token in ("--metadata", "--features", "--output-dir", "selected_detector.npz")
        )
        and all(
            token in scoring
            for token in (
                "--metadata",
                "--features",
                "--detector",
                "id_column",
                "metadata_json",
                "text_tokens",
                "text_representation",
            )
        )
    )
    _record(checks, "pipeline_entrypoint_contracts", entrypoints_ok)

    readme = (root / "README.md").read_text(encoding="utf-8")
    _record(
        checks,
        "pipeline_provenance_documented",
        "provenance" in readme.lower()
        and "aegis_llava_onevision_05b_text_detector_tuned_v3.npz" in readme,
    )


def _verify_redteam(checks: list[dict[str, object]]) -> None:
    root = ROOT / "red_teaming"
    actual_files = _relative_file_set(root)
    _record(
        checks,
        "redteam_current_tooling_layout",
        actual_files == REDTEAM_FILES,
        {
            "missing": sorted(REDTEAM_FILES - actual_files),
            "unexpected": sorted(actual_files - REDTEAM_FILES),
        },
    )

    catalog = _load_json(root / "attack_catalog.json")
    variants = catalog.get("variants", [])
    catalog_ok = (
        catalog.get("schema_version") == 1
        and catalog.get("safety_profile") == "redacted_non_operational"
        and len(variants) == 6
        and len({entry.get("name") for entry in variants}) == 6
        and all(entry.get("deterministic") is True for entry in variants)
        and all(entry.get("description") for entry in variants)
    )
    _record(checks, "redteam_attack_catalog", catalog_ok)

    schema = _load_json(root / "response_judgment_schema.json")
    required = {
        "sample_id",
        "attack_style",
        "condition",
        "attack_succeeded",
        "effective_action",
        "recommended_action",
        "traffic_mode",
        "downstream_disposition",
        "judge_type",
        "judge_version",
        "response_sha256",
    }
    schema_ok = (
        required <= set(schema.get("required", []))
        and isinstance(schema.get("allOf"), list)
        and len(schema["allOf"]) >= 2
        and isinstance(schema.get("properties"), dict)
    )
    _record(checks, "redteam_response_schema", schema_ok)

    generator = (root / "generate_variants.py").read_text(encoding="utf-8")
    adaptive = (root / "evaluate_adaptive.py").read_text(encoding="utf-8")
    response = (root / "evaluate_attack_success.py").read_text(encoding="utf-8")
    tools_ok = (
        '"--output"' in generator
        and "required=True" in generator
        and "detector_artifact_block_threshold" in adaptive
        and "detector_threshold" in adaptive
        and "scored variant IDs are not aligned" in adaptive
        and "traffic_mode" in adaptive
        and "response_sha256" in response
        and "effective_action" in response
    )
    _record(checks, "redteam_entrypoint_contracts", tools_ok)

    safety_text = (root / "SAFE_USE.md").read_text(encoding="utf-8").lower()
    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    _record(
        checks,
        "redteam_safety_documented",
        "authorization" in safety_text
        and "redact" in safety_text
        and "traffic mode" in readme,
    )


def _verify_ablation(checks: list[dict[str, object]]) -> None:
    root = ROOT / "ablation_report"
    actual_files = _relative_file_set(root)
    _record(
        checks,
        "ablation_current_tooling_layout",
        actual_files == ABLATION_FILES,
        {
            "missing": sorted(ABLATION_FILES - actual_files),
            "unexpected": sorted(actual_files - ABLATION_FILES),
        },
    )

    source = (root / "run_ablation.py").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    contract_ok = (
        '"--features"' in source
        and "required=True" in source
        and "run_manifest.json" in source
        and "preprocessing_sha256" in source
        and "scores_sha256" in source
        and "detector_sha256" in source
        and "features_sha256" in source
        and "does not describe the supplied inputs" in source
        and "provenance" in source
        and "provenance" in readme.lower()
        and "current" in readme.lower()
    )
    _record(checks, "ablation_entrypoint_contract", contract_ok)


def _verify_runtime(checks: list[dict[str, object]]) -> None:
    record_path = ROOT / "runtime_validation" / "docker_smoke_2026-07-31.json"
    record = _load_json(record_path)
    expected_keys = {
        "schema_version",
        "validation_date",
        "status",
        "scope",
        "package_version",
        "docker",
        "llava_artifact",
        "functional_validation",
        "gui",
        "docker_recheck",
        "teardown",
    }
    pyproject = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
    package_version = version_match.group(1) if version_match else None
    identity_ok = (
        set(record) == expected_keys
        and record["schema_version"] == 3
        and record["validation_date"] == "2026-07-31"
        and record["status"] == "pass"
        and record["package_version"] == package_version
    )
    _record(
        checks,
        "runtime_record_identity",
        identity_ok,
        {
            "schema_version": record.get("schema_version"),
            "package_version": package_version,
            "keys": sorted(record),
        },
    )

    artifact_record = record["llava_artifact"]
    artifact_relative = Path(artifact_record["path"])
    artifact_path = (REPOSITORY / artifact_relative).resolve()
    try:
        artifact_path.relative_to(REPOSITORY.resolve())
        artifact_path_safe = not artifact_relative.is_absolute()
    except ValueError:
        artifact_path_safe = False
    if not artifact_path_safe:
        raise ValueError("runtime detector path must remain inside the repository")

    with np.load(artifact_path, allow_pickle=False) as artifact:
        weights = np.asarray(artifact["weights"], dtype=np.float64)
        block_threshold = float(
            np.asarray(artifact["threshold"]).reshape(-1)[0]
        )
        pooling = str(np.asarray(artifact["pooling"]).reshape(-1)[0])
        layer = int(np.asarray(artifact["layer"]).reshape(-1)[0])
        source = str(np.asarray(artifact["source"]).reshape(-1)[0])
        model_revision = str(
            np.asarray(artifact["model_revision"]).reshape(-1)[0]
        )
        tokenizer_revision = str(
            np.asarray(artifact["tokenizer_revision"]).reshape(-1)[0]
        )
        preprocessing = str(
            np.asarray(artifact["preprocessing_sha256"]).reshape(-1)[0]
        )
    artifact_ok = (
        _raw_sha256(artifact_path) == RUNTIME_ARTIFACT_SHA256
        == artifact_record["sha256"]
        and artifact_record["feature_dim"] == weights.size == 896
        and artifact_record["pooling"] == pooling == "text_tokens"
        and artifact_record["layer"] == layer == -1
        and artifact_record["source"] == source
        and artifact_record["model_revision"] == model_revision
        and artifact_record["tokenizer_revision"] == tokenizer_revision
        and artifact_record["preprocessing_sha256"] == preprocessing
        and re.fullmatch(r"[0-9a-f]{64}", preprocessing) is not None
        and math.isclose(
            float(artifact_record["block_threshold"]),
            block_threshold,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        )
    )
    _record(
        checks,
        "runtime_tuned_v3_artifact_contract",
        artifact_ok,
        {
            "sha256": _raw_sha256(artifact_path),
            "feature_dim": weights.size,
            "pooling": pooling,
            "layer": layer,
        },
    )

    config = _load_json(
        REPOSITORY / "configs" / "aegis.llava.deployment.container.json"
    )
    config_ok = (
        config["detector"] == artifact_record["path"]
        and config["target_profile"] == "llava05b"
        and config["traffic_mode"] == "shadow"
        and config["policy"].get("block_threshold") is None
        and math.isclose(
            float(config["policy"]["review_threshold"]),
            float(artifact_record["review_threshold"]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        )
    )
    _record(checks, "runtime_config_record_alignment", config_ok)

    profile_text = (REPOSITORY / "AEGIS" / "target_profiles.py").read_text(
        encoding="utf-8"
    )
    profile_ok = all(
        token in profile_text
        for token in (
            "aegis_llava_onevision_05b_text_detector_tuned_v3.npz",
            '"feature_dim": 896',
            '"pooling": "text_tokens"',
            "review_threshold=0.23579741243702598",
            "c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c",
        )
    )
    _record(checks, "runtime_target_profile_alignment", profile_ok)

    docker = record["docker"]
    validation = record["functional_validation"]
    gui = record["gui"]
    teardown = record["teardown"]
    functional_ok = (
        docker["image"]["fresh_build_passed"] is True
        and validation["ready_http"] == 200
        and validation["ready_ok"] is True
        and validation["worker_ready"] is True
        and validation["worker_restarts"] == 0
        and validation["authentication"]
        == {
            "unauthenticated_guard_http": 401,
            "authenticated_metrics_http": 200,
            "authenticated_guard_http": 200,
        }
        and validation["guard_smoke"]["action"] == "allow"
        and validation["guard_smoke"]["recommended_action"] == "allow"
        and validation["guard_smoke"]["modality"] == "image_text"
        and validation["traffic_mode_round_trip"] == ["shadow", "review", "shadow"]
        and gui["home_http"] == 200
        and gui["evaluate_http"] == 200
        and gui["shutdown_http"] == 202
        and gui["process_exit_code"] == 0
        and teardown["temporary_validation_container_removed"] is True
        and teardown["port_8766_listening"] is False
        and teardown["port_8767_listening"] is False
    )
    _record(checks, "runtime_functional_record_complete", functional_ok)

    recheck = record["docker_recheck"]
    recheck_ok = (
        recheck["docker_engine_running"] is True
        and recheck["container_log_observations"]["authentication_required"] is True
        and recheck["container_log_observations"]["traffic_mode"] == "shadow"
        and recheck["container_log_observations"]["warmup_completed"] is True
        and recheck["container_log_observations"]["health_checks_passed"] == 5
        and recheck["container_log_observations"]["exit_code"] == 0
        and recheck["ports_after_exit"]["8766_listening"] is False
        and recheck["ports_after_exit"]["8767_listening"] is False
    )
    _record(checks, "runtime_docker_recheck", recheck_ok)

    runtime_readme = (
        ROOT / "runtime_validation" / "README.md"
    ).read_text(encoding="utf-8")
    deliverables_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    _record(
        checks,
        "runtime_evidence_scope_documented",
        "functional smoke test" in runtime_readme.lower()
        and "not an accuracy, robustness, latency" in runtime_readme.lower()
        and "896-dimensional" in deliverables_readme
        and "aegis_llava_onevision_05b_text_detector_tuned_v3.npz"
        in deliverables_readme,
    )


def _verify_source_files(checks: list[dict[str, object]]) -> None:
    canonical_files = list(_iter_canonical_files())
    syntax_errors: list[str] = []
    json_errors: list[str] = []
    for path in canonical_files:
        if path.suffix == ".py":
            try:
                ast.parse(
                    path.read_text(encoding="utf-8-sig"),
                    filename=str(path),
                )
            except SyntaxError as exc:
                syntax_errors.append(
                    f"{path.relative_to(ROOT).as_posix()}:{exc.lineno}"
                )
        elif path.suffix == ".json":
            try:
                _load_json(path)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                json_errors.append(
                    f"{path.relative_to(ROOT).as_posix()}:{exc}"
                )
    _record(
        checks,
        "source_python_syntax",
        not syntax_errors,
        syntax_errors or sum(path.suffix == ".py" for path in canonical_files),
    )
    _record(
        checks,
        "source_json_parse",
        not json_errors,
        json_errors or sum(path.suffix == ".json" for path in canonical_files),
    )

    link_errors: list[str] = []
    for path in canonical_files:
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8-sig")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            clean = target.strip().strip("<>").split("#", 1)[0]
            if (
                not clean
                or clean.startswith(("#", "http://", "https://", "mailto:"))
            ):
                continue
            candidate = (path.parent / clean).resolve()
            try:
                candidate.relative_to(REPOSITORY.resolve())
            except ValueError:
                link_errors.append(
                    f"{path.relative_to(ROOT).as_posix()}:{target}"
                )
                continue
            if not candidate.exists():
                link_errors.append(
                    f"{path.relative_to(ROOT).as_posix()}:{target}"
                )
    _record(checks, "source_local_markdown_links", not link_errors, link_errors)

    ci_entry = (ROOT / "ci" / "run_checks.py").read_text(encoding="utf-8")
    _record(
        checks,
        "source_ci_read_only_verifier",
        "--check" in ci_entry and "unittest" in ci_entry,
    )


def _verify_stored_manifest(
    checks: list[dict[str, object]],
    current: dict[str, object],
) -> None:
    try:
        stored = _load_json(MANIFEST_PATH)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        _record(
            checks,
            "manifest_exact_match",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        return

    files = stored.get("files", [])
    paths = [entry.get("path") for entry in files if isinstance(entry, dict)]
    structure_ok = (
        stored.get("schema_version") == MANIFEST_SCHEMA_VERSION
        and stored.get("root") == "deliverables"
        and paths == sorted(paths)
        and len(paths) == len(set(paths))
        and stored.get("file_count") == len(files)
        and stored.get("total_bytes")
        == sum(int(entry["bytes"]) for entry in files)
    )
    _record(
        checks,
        "manifest_structure",
        structure_ok,
        {
            "schema_version": stored.get("schema_version"),
            "file_count": stored.get("file_count"),
        },
    )
    _record(
        checks,
        "manifest_exact_match",
        stored == current,
        {
            "stored_artifact_set": stored.get("artifact_set_sha256"),
            "current_artifact_set": current["artifact_set_sha256"],
            "stored_files": stored.get("file_count"),
            "current_files": current["file_count"],
        },
    )


def _verify_stored_report(
    checks: list[dict[str, object]],
    current_manifest: dict[str, object],
) -> None:
    try:
        report = _load_json(REPORT_PATH)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        _record(
            checks,
            "verification_report_consistency",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        return
    runtime = _load_json(
        ROOT / "runtime_validation" / "docker_smoke_2026-07-31.json"
    )
    report_ok = (
        report.get("schema_version") == REPORT_SCHEMA_VERSION
        and report.get("status") == "pass"
        and report.get("scope") == "current_aegis_deliverables"
        and report.get("package_version") == runtime["package_version"]
        and report.get("runtime_validation_date") == runtime["validation_date"]
        and report.get("artifact_set_sha256")
        == current_manifest["artifact_set_sha256"]
        and report.get("manifest_sha256") == _raw_sha256(MANIFEST_PATH)
        and report.get("manifest_files") == current_manifest["file_count"]
        and report.get("manifest_total_canonical_bytes")
        == current_manifest["total_bytes"]
        and report.get("checks_passed") == report.get("checks_total")
    )
    _record(
        checks,
        "verification_report_consistency",
        report_ok,
        {
            "status": report.get("status"),
            "artifact_set_sha256": report.get("artifact_set_sha256"),
        },
    )


def _build_manifest() -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for path in _iter_canonical_files():
        payload, hash_mode = _canonical_payload(path)
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "hash_mode": hash_mode,
            }
        )
    entries.sort(key=lambda entry: str(entry["path"]))
    artifact_set = hashlib.sha256(
        json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "root": "deliverables",
        "hash_algorithm": "sha256",
        "text_hash_mode": "utf8_bom_removed_newlines_normalized_to_lf",
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "artifact_set_sha256": artifact_set,
        "files": entries,
    }


def _build_stored_report(
    checks: list[dict[str, object]],
    manifest: dict[str, object],
) -> dict[str, object]:
    runtime = _load_json(
        ROOT / "runtime_validation" / "docker_smoke_2026-07-31.json"
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "pass" if _all_passed(checks) else "fail",
        "scope": "current_aegis_deliverables",
        "package_version": runtime["package_version"],
        "runtime_validation_date": runtime["validation_date"],
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "verifier_version": VERIFIER_VERSION,
        "artifact_set_sha256": manifest["artifact_set_sha256"],
        "manifest_sha256": _raw_sha256(MANIFEST_PATH),
        "manifest_files": manifest["file_count"],
        "manifest_total_canonical_bytes": manifest["total_bytes"],
        "checks_passed": sum(bool(check["passed"]) for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "toolchain": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def _print_result(
    mode: str,
    checks: list[dict[str, object]],
    manifest: dict[str, object],
    stored_report: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "mode": mode,
        "status": "pass" if _all_passed(checks) else "fail",
        "checks_passed": sum(bool(check["passed"]) for check in checks),
        "checks_total": len(checks),
        "artifact_set_sha256": manifest["artifact_set_sha256"],
        "manifest_files": manifest["file_count"],
        "checks": checks,
    }
    if stored_report is not None:
        payload["verification_report"] = {
            "generated_at_utc": stored_report["generated_at_utc"],
            "manifest_sha256": stored_report["manifest_sha256"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _iter_canonical_files():
    for directory, names, filenames in os.walk(ROOT, followlinks=False):
        names[:] = sorted(
            name for name in names if name not in EXCLUDED_DIRECTORY_NAMES
        )
        for filename in sorted(filenames):
            if (
                filename in EXCLUDED_FILE_NAMES
                or filename.endswith((".pyc", ".pyo", ".tmp"))
                or filename.endswith(".xlsx.inspect.ndjson")
            ):
                continue
            yield Path(directory) / filename


def _canonical_payload(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return raw, "raw"
    text = raw.decode("utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.encode("utf-8"), "normalized_lf"


def _relative_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED_FILE_NAMES
        and path.suffix not in {".pyc", ".pyo"}
        and not any(part in EXCLUDED_DIRECTORY_NAMES for part in path.parts)
    }


def _record(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    detail: object | None = None,
) -> None:
    item: dict[str, object] = {"name": name, "passed": bool(passed)}
    if detail is not None:
        item["detail"] = detail
    checks.append(item)


def _all_passed(checks: list[dict[str, object]]) -> bool:
    return all(bool(check["passed"]) for check in checks)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


if __name__ == "__main__":
    main()
