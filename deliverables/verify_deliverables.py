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
import subprocess
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
VERIFIER_VERSION = "2.0.0"
REQUIRED_DIRECTORIES = {
    "ablation_report",
    "benchmark",
    "ci",
    "pipeline",
    "red_teaming",
    "runtime_validation",
}
ALLOWED_TOP_LEVEL_FILES = {
    "HISTORICAL_SNAPSHOT.md",
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
FROZEN_HASHES = {
    "pipeline/fixtures/smoke_features.npz": (
        "93cd720edfc6bfbd55a94eba05dbe3079a3d0e350d390ac8ebe5273b38fa32a2"
    ),
    "pipeline/artifacts/smoke/selected_detector.npz": (
        "c008be65092b3b113832d91c968cf4c3842745c78f38544b4cede17468ba7b71"
    ),
    "red_teaming/generated/variants.csv": (
        "c807e6d12676a367783165ecd950bff8c2eca7f3b357e8e3dd47a8c0c5458ba7"
    ),
    "red_teaming/generated/variant_smoke_features.npz": (
        "f6a3ba6d712289b395737041c4fa25afebbb1ed3d4bd053a509a239fb5fe2b5c"
    ),
    "red_teaming/generated/scored_variants.csv": (
        "5a0b3dc9830317c223b2da9f62551b326216f42e96fb384dabcf3484475d5950"
    ),
}
RUNTIME_ARTIFACT_SHA256 = (
    "9816ed0b706b3d0e01162701310ad77bc795c7dd99f7e2e6da4133d6fa299587"
)
VALIDATED_OPERATIONAL_SOURCE_COMMIT = (
    "71a183bd10a102324a7c782155075e977d0e83ae"
)
VALIDATION_EXECUTION_COMMIT_BEFORE_SQUASH = (
    "b0a4aca0eadfdf61e2eecd615de0417689729e00"
)
VALIDATION_EXECUTION_TREE_BEFORE_SQUASH = (
    "23673aaa080320a5ba919b7229d5206291d28378"
)
FLOAT_TOLERANCE = 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the restored AEGIS Final Report evidence without modifying it."
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
        except Exception as exc:  # keep failures structured for CI and report review
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
    required_patterns = (
        "deliverables/ablation_report/results/** -text",
        "deliverables/pipeline/artifacts/smoke/** -text",
        "deliverables/red_teaming/generated/** -text",
        "*.npz binary",
        "*.xlsx binary",
    )
    _record(
        checks,
        "scope_frozen_byte_identity_attributes",
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
        workbook_xml = archive.read("xl/workbook.xml")
        workbook_root = ElementTree.fromstring(workbook_xml)
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
    benchmark = _read_csv(ROOT / "benchmark" / "data" / "benchmark.csv")
    bundle_path = root / "fixtures" / "smoke_features.npz"
    required_bundle_keys = {
        "sample_ids",
        "text_embeddings",
        "image_embeddings",
        "attribution_features",
        "feature_source",
    }
    with np.load(bundle_path, allow_pickle=False) as bundle:
        keys = set(bundle.files)
        ids = np.asarray(bundle["sample_ids"]).astype(str)
        text = np.asarray(bundle["text_embeddings"], dtype=np.float64)
        image = np.asarray(bundle["image_embeddings"], dtype=np.float64)
        attribution = np.asarray(
            bundle["attribution_features"], dtype=np.float64
        )
        source = str(np.asarray(bundle["feature_source"]).reshape(-1)[0])
    _record(
        checks,
        "pipeline_feature_bundle_contract",
        required_bundle_keys <= keys
        and text.shape == (64, 32)
        and image.shape == (64, 32)
        and attribution.shape == (64, 8)
        and all(np.all(np.isfinite(array)) for array in (text, image, attribution))
        and len(set(ids)) == len(ids)
        and np.array_equal(
            ids,
            np.asarray([row["sample_id"] for row in benchmark]),
        )
        and source == "deterministic_smoke_fixture_not_mllm_evidence",
        {
            "text": list(text.shape),
            "image": list(image.shape),
            "attribution": list(attribution.shape),
            "source": source,
        },
    )

    detector_path = root / "artifacts" / "smoke" / "selected_detector.npz"
    with np.load(detector_path, allow_pickle=False) as detector:
        detector_keys = set(detector.files)
        weights = np.asarray(detector["weights"], dtype=np.float64)
        mean = np.asarray(detector["mean"], dtype=np.float64)
        scale = np.asarray(detector["scale"], dtype=np.float64)
        bias = np.asarray(detector["bias"], dtype=np.float64)
        threshold = float(np.asarray(detector["threshold"]).reshape(-1)[0])
        metadata = json.loads(
            str(np.asarray(detector["metadata_json"]).reshape(-1)[0])
        )
    required_detector_keys = {
        "weights",
        "mean",
        "scale",
        "bias",
        "threshold",
        "metadata_json",
    }
    detector_ok = (
        required_detector_keys <= detector_keys
        and weights.shape == mean.shape == scale.shape == (8,)
        and bias.size == 1
        and all(
            np.all(np.isfinite(array))
            for array in (weights, mean, scale, bias)
        )
        and np.all(scale > 0)
        and math.isclose(
            threshold,
            0.8202568457258909,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        )
        and metadata.get("feature_view") == "attribution"
        and metadata.get("feature_source") == source
    )
    _record(
        checks,
        "pipeline_fixture_detector_contract",
        detector_ok,
        {"feature_dim": weights.size, "threshold": threshold, **metadata},
    )

    predictions = _read_csv(
        root / "artifacts" / "smoke" / "test_predictions.csv"
    )
    test_ids = [
        row["sample_id"] for row in benchmark if row["split"] == "test"
    ]
    predictions_ok = (
        len(predictions) == 16
        and [row["sample_id"] for row in predictions] == test_ids
        and all(0.0 <= float(row["risk_score"]) <= 1.0 for row in predictions)
        and all(
            math.isclose(
                float(row["threshold"]),
                threshold,
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
            for row in predictions
        )
    )
    _record(checks, "pipeline_test_prediction_alignment", predictions_ok)

    summary = _load_json(root / "artifacts" / "smoke" / "summary.json")
    _record(
        checks,
        "pipeline_historical_scope_disclaimer",
        summary["feature_source"] == source
        and summary["selection_rule"].startswith("maximum validation")
        and "software" in summary["limitations"],
    )
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    _record(
        checks,
        "pipeline_reproduction_dependencies_pinned",
        requirements.splitlines() == ["numpy==2.3.5", "Pillow==12.2.0"],
    )
    feature_schema = _load_json(
        root / "schemas" / "feature_bundle.schema.json"
    )
    _record(
        checks,
        "pipeline_feature_schema_provenance",
        "provenance" in feature_schema["properties"]
        and feature_schema["properties"]["text_embeddings"]["type"] == "array",
    )
    _verify_frozen_hashes(checks, prefix="pipeline")


def _verify_redteam(checks: list[dict[str, object]]) -> None:
    root = ROOT / "red_teaming"
    generated = root / "generated"
    variants = _read_csv(generated / "variants.csv")
    scored = _read_csv(generated / "scored_variants.csv")

    groups = Counter(row["base_sample_id"] for row in variants)
    query_groups: dict[str, list[int]] = defaultdict(list)
    for row in variants:
        query_groups[row["base_sample_id"]].append(int(row["query_index"]))
    _record(
        checks,
        "redteam_fixed_panel_shape",
        len(variants) == 48
        and len(groups) == 8
        and set(groups.values()) == {6}
        and all(sorted(values) == list(range(1, 7)) for values in query_groups.values())
        and len({row["variant_id"] for row in variants}) == 48
        and len({row["variant_type"] for row in variants}) == 6,
        {"rows": len(variants), "base_samples": len(groups)},
    )

    content_errors: list[str] = []
    relocated_paths = 0
    for row in variants:
        normalized = " ".join(row["prompt_text"].lower().split())
        if hashlib.sha256(normalized.encode("utf-8")).hexdigest() != row[
            "prompt_sha256"
        ]:
            content_errors.append(f"{row['variant_id']}:prompt_hash")
        if row.get("safe_handling") != "operational_details_redacted":
            content_errors.append(f"{row['variant_id']}:safe_handling")
        if row["image_path"]:
            image_path, relocated = _resolve_historical_redteam_image(
                generated,
                row["image_path"],
            )
            relocated_paths += int(relocated)
            if (
                not image_path.is_file()
                or _raw_sha256(image_path) != row["image_sha256"]
            ):
                content_errors.append(f"{row['variant_id']}:image_hash")
    _record(
        checks,
        "redteam_hashes_and_historical_relocation",
        not content_errors and relocated_paths == 40,
        content_errors or {"relocated_paths": relocated_paths},
    )

    scored_ok = (
        len(scored) == len(variants)
        and [row["variant_id"] for row in scored]
        == [row["variant_id"] for row in variants]
        and all(0.0 <= float(row["risk_score"]) <= 1.0 for row in scored)
        and {row["feature_view"] for row in scored} == {"attribution"}
        and all(
            math.isclose(
                float(row["detector_threshold"]),
                0.8202568457258909,
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
            for row in scored
        )
    )
    _record(checks, "redteam_scored_panel_alignment", scored_ok)

    adaptive_payload = _load_json(generated / "adaptive_summary.json")
    expected_summary = _recompute_best_of_n(scored, [1, 3, 6])
    actual_summary = adaptive_payload.get("summary", [])
    adaptive_ok = len(actual_summary) == len(expected_summary) and all(
        int(actual["query_budget"]) == int(expected["query_budget"])
        and int(actual["n_samples"]) == int(expected["n_samples"])
        and all(
            math.isclose(
                float(actual[name]),
                float(expected[name]),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
            for name in (
                "threshold",
                "malicious_recall",
                "evasion_rate",
                "mean_worst_case_score",
            )
        )
        for actual, expected in zip(actual_summary, expected_summary)
    )
    _record(
        checks,
        "redteam_best_of_n_recomputed",
        adaptive_ok,
        expected_summary,
    )

    schema = _load_json(root / "response_judgment_schema.json")
    _record(
        checks,
        "redteam_response_schema_runtime_conditions",
        isinstance(schema.get("allOf"), list) and len(schema["allOf"]) >= 2,
    )
    safety_text = (root / "SAFE_USE.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    _record(
        checks,
        "redteam_evidence_boundary_documented",
        "bounded best-of-N" in readme
        and "shadow" in readme
        and "Never overwrite" in safety_text,
    )
    _verify_frozen_hashes(checks, prefix="redteam")


def _verify_ablation(checks: list[dict[str, object]]) -> None:
    root = ROOT / "ablation_report"
    results = root / "results"
    expected_rows = {
        "feature_signal_ablation.csv": 7,
        "low_label_seed_runs.csv": 20,
        "low_label_summary.csv": 4,
        "attack_family_transfer.csv": 16,
        "uncertainty_coverage.csv": 4,
        "adaptive_redteam_summary.csv": 3,
    }
    row_counts = {
        name: len(_read_csv(results / name)) for name in expected_rows
    }
    _record(
        checks,
        "ablation_result_table_shapes",
        row_counts == expected_rows,
        row_counts,
    )

    numeric_errors: list[str] = []
    for name in expected_rows:
        rows = _read_csv(results / name)
        for row_number, row in enumerate(rows, start=2):
            for column, value in row.items():
                if value == "" or not _looks_numeric(value):
                    continue
                if not math.isfinite(float(value)):
                    numeric_errors.append(f"{name}:{row_number}:{column}")
    _record(
        checks,
        "ablation_numeric_results_finite",
        not numeric_errors,
        numeric_errors,
    )

    adaptive_csv = _read_csv(results / "adaptive_redteam_summary.csv")
    adaptive_json = _load_json(
        ROOT / "red_teaming" / "generated" / "adaptive_summary.json"
    )["summary"]
    adaptive_alignment = len(adaptive_csv) == len(adaptive_json) and all(
        int(csv_row["query_budget"]) == int(json_row["query_budget"])
        and all(
            math.isclose(
                float(csv_row[name]),
                float(json_row[name]),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
            for name in (
                "threshold",
                "malicious_recall",
                "evasion_rate",
                "mean_worst_case_score",
            )
        )
        for csv_row, json_row in zip(adaptive_csv, adaptive_json)
    )
    _record(
        checks,
        "ablation_adaptive_table_alignment",
        adaptive_alignment,
    )

    report = (root / "REPORT.md").read_text(encoding="utf-8")
    required_sections = (
        "## Evidence status",
        "## Relationship to the current runtime",
        "## Signal importance",
        "## Low-label and pseudo-label ablation",
        "## Attack-family transfer",
        "## Uncertainty and review coverage",
        "## Bounded best-of-N fixture evasion",
        "## Legacy MLLM evidence",
        "## Reproduction",
    )
    report_ok = (
        all(section in report for section in required_sections)
        and RUNTIME_ARTIFACT_SHA256 in report
        and "not evidence of MLLM safety performance" in report
        and "not a sequential" in report
        and "reproductions/ablation_v1" in report
    )
    _record(checks, "ablation_report_evidence_boundaries", report_ok)

    historical_manifest = _load_json(results / "run_manifest.json")
    snapshot = (ROOT / "HISTORICAL_SNAPSHOT.md").read_text(encoding="utf-8")
    _record(
        checks,
        "ablation_historical_provenance_mapped",
        historical_manifest["rows"] == 64
        and "annotated_benchmark" in historical_manifest["metadata"]
        and "absolute Windows paths" in snapshot,
    )


def _verify_runtime(checks: list[dict[str, object]]) -> None:
    record_path = (
        ROOT / "runtime_validation" / "docker_smoke_2026-07-31.json"
    )
    record = _load_json(record_path)
    pyproject = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(
        r"(?m)^version\s*=\s*\"([^\"]+)\"",
        pyproject,
    )
    package_version = version_match.group(1) if version_match else None
    repository_record = record["repository"]
    validated_commit = repository_record[
        "validated_tracked_operational_source_commit"
    ]
    execution_commit_before_squash = repository_record[
        "validation_execution_commit_before_squash"
    ]
    execution_tree_before_squash = repository_record[
        "validation_execution_tree_before_squash"
    ]
    execution_parent_commit = repository_record[
        "validation_execution_parent_commit"
    ]
    runtime_scope = repository_record[
        "validated_tracked_operational_scope"
    ]
    changed_build_input = repository_record[
        "changed_non_executable_build_input"
    ]
    runtime_scope_valid = (
        isinstance(runtime_scope, list)
        and bool(runtime_scope)
        and all(
            isinstance(path, str)
            and path
            and "\\" not in path
            and not Path(path).is_absolute()
            and ".." not in Path(path).parts
            for path in runtime_scope
        )
    )
    _record(
        checks,
        "runtime_record_identity",
        record["schema_version"] == 2
        and record["validation_date"] == "2026-07-31"
        and repository_record["package_version"] == package_version
        and re.fullmatch(r"[0-9a-f]{40}", validated_commit) is not None
        and validated_commit == VALIDATED_OPERATIONAL_SOURCE_COMMIT
        and execution_commit_before_squash
        == VALIDATION_EXECUTION_COMMIT_BEFORE_SQUASH
        and execution_tree_before_squash
        == VALIDATION_EXECUTION_TREE_BEFORE_SQUASH
        and execution_parent_commit == VALIDATED_OPERATIONAL_SOURCE_COMMIT
        and execution_commit_before_squash
        in repository_record["history_note"]
        and validated_commit in repository_record["history_note"]
        and runtime_scope_valid,
        {
            "package_version": package_version,
            "validated_tracked_operational_source_commit": (
                validated_commit
            ),
            "validation_execution_commit_before_squash": (
                execution_commit_before_squash
            ),
            "validation_execution_tree_before_squash": (
                execution_tree_before_squash
            ),
            "validation_execution_parent_commit": (
                execution_parent_commit
            ),
            "validated_tracked_operational_scope_paths": len(runtime_scope),
        },
    )

    commit_exists = False
    try:
        completed = subprocess.run(
            ["git", "cat-file", "-e", f"{validated_commit}^{{commit}}"],
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit_exists = completed.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        commit_exists = False
    _record(
        checks,
        "runtime_operational_source_commit_exists",
        commit_exists,
    )

    runtime_scope_aligned = False
    if commit_exists and runtime_scope_valid:
        try:
            completed = subprocess.run(
                [
                    "git",
                    "diff",
                    "--quiet",
                    validated_commit,
                    "--",
                    *runtime_scope,
                ],
                cwd=REPOSITORY,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            runtime_scope_aligned = completed.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            runtime_scope_aligned = False
    _record(
        checks,
        "runtime_tracked_operational_scope_alignment",
        runtime_scope_aligned,
        {
            "anchor": validated_commit,
            "paths": runtime_scope,
        },
    )
    _record(
        checks,
        "runtime_source_scope_qualifications",
        changed_build_input
        == {
            "path": "README.md",
            "copied_by_dockerfile": True,
            "execution_effect": (
                "changes documentation, package metadata, and reproducible "
                "image bytes, but not AEGIS executable modules, configs, or "
                "model artifacts"
            ),
            "note": (
                "The root README changed during deliverable cleanup and "
                "restoration, so it is deliberately excluded from the "
                "byte-identity claim. The recorded Docker image ID belongs "
                "to the pre-squash execution and is not a promised rebuild "
                "identity at the surviving source anchor."
            ),
        }
        and "tracked files only"
        in repository_record["untracked_model_cache_note"]
        and "recorded model/tokenizer revisions"
        in repository_record["untracked_model_cache_note"],
    )

    config = _load_json(
        REPOSITORY / "configs" / "aegis.llava.deployment.container.json"
    )
    artifact_record = record["llava_artifact"]
    artifact_path = REPOSITORY / artifact_record["path"]
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
        and math.isclose(
            artifact_record["block_threshold"],
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
            '"min_available_physical_bytes": 2147483648',
            "review_threshold=0.23579741243702598",
            "c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c",
        )
    )
    _record(checks, "runtime_target_profile_alignment", profile_ok)

    smoke = record["ephemeral_validation"]
    gui = record["gui"]
    functional_ok = (
        record["docker"]["image"]["fresh_build_passed"] is True
        and smoke["ready_http"] == 200
        and smoke["authentication"] == {
            "unauthenticated_guard_http": 401,
            "authenticated_metrics_http": 200,
            "authenticated_guard_http": 200,
        }
        and smoke["guard_smoke"]["action"] == "allow"
        and smoke["guard_smoke"]["recommended_action"] == "allow"
        and smoke["traffic_mode_round_trip"] == ["shadow", "review", "shadow"]
        and gui["home_http"] == 200
        and gui["evaluate_http"] == 200
        and gui["shutdown_http"] == 202
        and gui["process_exit_code"] == 0
        and record["teardown"]["temporary_validation_container_removed"] is True
        and record["teardown"]["port_8766_listening"] is False
        and record["teardown"]["port_8767_listening"] is False
    )
    _record(checks, "runtime_functional_record_complete", functional_ok)

    recheck = record["post_restoration_docker_recheck"]
    _record(
        checks,
        "runtime_post_restoration_docker_recheck",
        recheck["docker_engine_running"] is True
        and recheck["container_log_observations"]["authentication_required"] is True
        and recheck["container_log_observations"]["traffic_mode"] == "shadow"
        and recheck["container_log_observations"]["warmup_completed"] is True
        and recheck["container_log_observations"]["health_checks_passed"] == 5
        and recheck["container_log_observations"]["exit_code"] == 0
        and recheck["ports_after_exit"]["8766_listening"] is False
        and recheck["ports_after_exit"]["8767_listening"] is False
        and "did not repeat" in recheck["scope_note"],
    )

    stock = record["stock_compose_profile"]
    resource_ok = (
        stock["failure_kind"] == "resource_preflight"
        and stock["oom_killed"] is False
        and stock["required_available_physical_bytes"] == 2 * 1024**3
        and stock["observed_available_physical_bytes_min"]
        < stock["required_available_physical_bytes"]
        and all(stock["passed_checks"].values())
    )
    _record(checks, "runtime_resource_constraint_qualified", resource_ok)

    runtime_readme = (
        ROOT / "runtime_validation" / "README.md"
    ).read_text(encoding="utf-8")
    deliverables_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    _record(
        checks,
        "runtime_evidence_limits_documented",
        "not an accuracy, robustness, latency" in runtime_readme
        and "validated_tracked_operational_source_commit"
        in runtime_readme
        and "README.md" in runtime_readme
        and "tracked-path comparison" in runtime_readme
        and re.search(r"former\s+execution hash", runtime_readme) is not None
        and VALIDATION_EXECUTION_COMMIT_BEFORE_SQUASH in runtime_readme
        and VALIDATED_OPERATIONAL_SOURCE_COMMIT in runtime_readme
        and "byte-identical" in runtime_readme
        and "896-dimensional detector deployed" in deliverables_readme,
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

    allowed_historical_references = {
        "HISTORICAL_SNAPSHOT.md",
        "red_teaming/README.md",
        "verify_deliverables.py",
    }
    stale_references: list[str] = []
    for path in canonical_files:
        relative = path.relative_to(ROOT).as_posix()
        if (
            path.suffix not in {".md", ".mjs", ".py", ".yml", ".yaml"}
            or relative in allowed_historical_references
        ):
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "annotated_benchmark" in text or "reproducible_pipeline" in text:
            stale_references.append(relative)
    _record(
        checks,
        "source_no_stale_executable_paths",
        not stale_references,
        stale_references,
    )

    ci_entry = (ROOT / "ci" / "run_checks.py").read_text(encoding="utf-8")
    _record(
        checks,
        "source_ci_read_only_verifier",
        "--check" in ci_entry and "unittest" in ci_entry,
    )

    reproduction = _load_json(
        ROOT / "ci" / "reproduction_validation_2026-07-31.json"
    )
    best_of_n = reproduction["research_workflow"]["best_of_n"]
    _record(
        checks,
        "source_reproduction_validation_record",
        reproduction["status"] == "pass"
        and reproduction["benchmark_rebuild"]["image_hash_mismatches"] == 0
        and reproduction["workbook_review"]["formula_error_matches"] == 0
        and reproduction["workbook_review"]["all_sheets_visually_reviewed"] is True
        and [row["query_budget"] for row in best_of_n] == [1, 3, 6]
        and [row["malicious_recall"] for row in best_of_n]
        == [0.75, 0.375, 0.0]
        and reproduction["research_workflow"][
            "historical_artifacts_overwritten"
        ]
        is False
        and reproduction["cleanup"]["temporary_reproduction_outputs_removed"]
        is True,
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
    report_ok = (
        report.get("schema_version") == 2
        and report.get("status") == "pass"
        and report.get("artifact_set_sha256")
        == current_manifest["artifact_set_sha256"]
        and report.get("manifest_sha256") == _raw_sha256(MANIFEST_PATH)
        and report.get("checks_passed") == report.get("checks_total")
        and report.get("validated_tracked_operational_source_commit")
        == _load_json(
            ROOT / "runtime_validation" / "docker_smoke_2026-07-31.json"
        )["repository"]["validated_tracked_operational_source_commit"]
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
    runtime_record = _load_json(
        ROOT / "runtime_validation" / "docker_smoke_2026-07-31.json"
    )
    return {
        "schema_version": 2,
        "status": "pass" if _all_passed(checks) else "fail",
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "verifier_version": VERIFIER_VERSION,
        "validated_tracked_operational_source_commit": (
            runtime_record["repository"][
                "validated_tracked_operational_source_commit"
            ]
        ),
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
            "platform": platform.platform(),
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


def _verify_frozen_hashes(
    checks: list[dict[str, object]],
    *,
    prefix: str,
) -> None:
    selected = {
        path: expected
        for path, expected in FROZEN_HASHES.items()
        if path.startswith("pipeline/")
        if prefix == "pipeline"
    }
    if prefix == "redteam":
        selected = {
            path: expected
            for path, expected in FROZEN_HASHES.items()
            if path.startswith("red_teaming/")
        }
    mismatches = {
        path: {
            "expected": expected,
            "actual": _raw_sha256(ROOT / path),
        }
        for path, expected in selected.items()
        if _raw_sha256(ROOT / path) != expected
    }
    _record(
        checks,
        f"{prefix}_frozen_byte_identities",
        not mismatches,
        mismatches or sorted(selected),
    )


def _resolve_historical_redteam_image(
    generated_root: Path,
    value: str,
) -> tuple[Path, bool]:
    normalized = value.replace("\\", "/")
    relocated = "../../annotated_benchmark/" in normalized
    if relocated:
        normalized = normalized.replace(
            "../../annotated_benchmark/",
            "../../benchmark/",
        )
    path = (generated_root / normalized).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"red-team image path escapes deliverables: {value}") from exc
    return path, relocated


def _recompute_best_of_n(
    scored: list[dict[str, str]],
    budgets: list[int],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scored:
        groups[row["base_sample_id"]].append(row)
    threshold = float(scored[0]["detector_threshold"])
    summary: list[dict[str, object]] = []
    for budget in budgets:
        selected_scores = []
        for variants in groups.values():
            eligible = sorted(
                variants,
                key=lambda row: int(row["query_index"]),
            )[:budget]
            selected_scores.append(
                min(float(row["risk_score"]) for row in eligible)
            )
        detected = sum(score >= threshold for score in selected_scores)
        summary.append(
            {
                "query_budget": budget,
                "n_samples": len(selected_scores),
                "threshold": threshold,
                "malicious_recall": detected / len(selected_scores),
                "evasion_rate": 1.0 - detected / len(selected_scores),
                "mean_worst_case_score": sum(selected_scores)
                / len(selected_scores),
            }
        )
    return summary


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


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    main()
