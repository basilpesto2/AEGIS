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
MANIFEST_SCHEMA_VERSION = 3
REPORT_SCHEMA_VERSION = 6
VERIFIER_VERSION = "4.5.0"
FLOAT_TOLERANCE = 1e-12
HISTORICAL_RUNTIME_RECORD = "v6_compose_validation_2026-08-29.json"
HISTORICAL_RUNTIME_RECORD_SHA256 = (
    "c3a48c987a6ae083361d4125b7541c0d19d5c29bcfcb882eb7240292fca4fcdf"
)
HISTORICAL_RUNTIME_ARTIFACT_SHA256 = {
    "llava05b": "eee5f82152042cc9b542b66810e5a1686244c954c680cef9638c69a266381aad",
    "qwen25vl3b": "38e12f4be0153039792b082867a74b954657d40f31c5013a67ab1444cb915e82",
}
HISTORICAL_RUNTIME_CONFIG_PATHS = {
    "llava05b": "configs/aegis.llava.deployment.container.json",
    "qwen25vl3b": "configs/aegis.deployment.container.json",
}
HISTORICAL_QWEN_RESOURCE_REQUIREMENTS = {
    "min_available_physical_bytes": 6_442_450_944,
    "min_available_virtual_bytes": 8_589_934_592,
    "min_cuda_device_memory_bytes": 7_516_192_768,
    "min_disk_free_bytes": 12_884_901_888,
    "min_model_cache_bytes": 6_442_450_944,
    "min_total_physical_bytes": 16_106_127_360,
}
CURRENT_RUNTIME_RECORD = "v7_compose_validation_2026-08-30.json"
CURRENT_RUNTIME_RECORD_SHA256 = (
    "0496760f8d69a1c4c306622b4d139b18302311231e87074c008435ead5d20b7c"
)
CURRENT_RUNTIME_IDENTITIES = {
    "llava05b": "25cebcf1e91125f4823176111d88cddae7379a0f70a1d8cc8af5ff07e22f08f6",
    "qwen25vl3b": "960f53c6fa3e584bfd2ff16a6ec3cd2d54f592e640e32da55748c2b8facd3a8e",
}
CURRENT_QWEN_PAIR_IDENTITY = (
    "7d44bf903ec880082aabc569b3468ef719746f87023422c0c721cf3463ff1174"
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
    "aegis_research/bordair_corpus.py",
    "aegis_research/bordair_corpus_prepare.py",
    "aegis_research/bordair_dual.py",
    "aegis_research/bordair_dual_evaluation.py",
    "aegis_research/bordair_dual_evaluation_validation.py",
    "aegis_research/bordair_dual_promotion.py",
    "aegis_research/bordair_dual_publish.py",
    "aegis_research/bordair_dual_qualification.py",
    "aegis_research/bordair_dual_training.py",
    "aegis_research/bordair_evaluation.py",
    "aegis_research/bordair_evaluation_validation.py",
    "aegis_research/bordair_paths.py",
    "aegis_research/bordair_promotion.py",
    "aegis_research/bordair_provenance.py",
    "aegis_research/bordair_training.py",
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
    "scripts/build_bordair_dual_pair_manifest.py",
    "scripts/evaluate_bordair_detector.py",
    "scripts/evaluate_bordair_dual_detector.py",
    "scripts/prepare_bordair_corpus.py",
    "scripts/prepare_bordair_dual_qualification.py",
    "scripts/publish_bordair_dual_pair.py",
    "scripts/run_experiment.py",
    "scripts/score_feature_bundle.py",
    "scripts/train_bordair_detector.py",
    "scripts/validate_bordair_corpus.py",
    "scripts/validate_bordair_dual_evaluation.py",
    "scripts/validate_bordair_dual_promotion.py",
    "scripts/validate_bordair_evaluation.py",
    "scripts/validate_bordair_promotion.py",
    "tests/test_bordair_dual_workflow.py",
    "tests/test_bordair_dual_qualification.py",
    "tests/test_bordair_dual_publish.py",
    "tests/test_bordair_durable_workflow.py",
    "tests/test_bordair_evaluation.py",
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
        and set(properties["pooling"].get("enum", []))
        == {"text_tokens", "image_tokens", "text_image_tokens"}
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
    signals = (root / "aegis_research" / "signals.py").read_text(
        encoding="utf-8"
    )
    entrypoints_ok = (
        all(
            token in extraction
            for token in (
                "--model-family",
                "--model-revision",
                "--tokenizer-revision",
                "--pooling",
                "--max-pixels",
                "--output-dir",
                "preprocessing_sha256",
                "variant_id",
                "image_tokens",
                "text_image_tokens",
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
            )
        )
        and all(
            token in signals
            for token in (
                "text_tokens",
                "text_representation",
                "text_image_tokens",
                "joint_representation",
            )
        )
    )
    _record(checks, "pipeline_entrypoint_contracts", entrypoints_ok)

    readme = (root / "README.md").read_text(encoding="utf-8")
    qualification = (
        root / "aegis_research" / "bordair_dual_qualification.py"
    ).read_text(encoding="utf-8")
    qualification_cli = (
        root / "scripts" / "prepare_bordair_dual_qualification.py"
    ).read_text(encoding="utf-8")
    qualification_tests = (
        root / "tests" / "test_bordair_dual_qualification.py"
    ).read_text(encoding="utf-8")
    qualification_contract = {
        "orchestrator": all(
            token in qualification
            for token in (
                'QUALIFICATION_EXECUTION = "atomic_extract_evaluate"',
                'WORKSPACE_PARENT = "frozen_evaluation_once_generic_v4"',
                "os.O_EXCL",
                "_fsync_directory",
                "_snapshot_implementation",
                "_copy_declared_inputs",
                "_model_snapshot_binding",
                "qualification_subject_sha256",
                "--qualification-subject",
                "--development-workspace",
                "current_runtime_match=False",
                "validate_generic_derived_evidence",
            )
        ),
        "thin_cli": all(
            token in qualification_cli
            for token in (
                "from aegis_research.bordair_dual_qualification import main",
                'if __name__ == "__main__":',
            )
        ),
        "mutation_tests": all(
            token in qualification_tests
            for token in (
                "test_canonical_subject_claim_prevents_redeclaration_if_workspace_moves",
                "test_strict_generic_package_builds_promotion_binding",
                "test_strict_generic_binding_allows_different_verifier_runtime_versions",
                "test_strict_generic_binding_rejects_tampered_recorded_runtime_versions",
                "test_model_snapshot_change_during_extraction_consumes_attempt",
            )
        ),
        "documentation": all(
            token in readme
            for token in (
                "Qwen v7 one-shot qualification",
                "prepare_bordair_dual_qualification.py",
                "frozen_evaluation_once_generic_v4",
                "--qualification-subject",
                "fail-closed local ledger",
                "not external attestation",
                "exact inspected Docker",
                "--dual-qualification-evidence-dir",
                "--runtime-image-id",
            )
        )
        and readme.count("\n  $imageId `\n") >= 6
        and "\n  $image `\n" not in readme,
    }
    _record(
        checks,
        "pipeline_one_shot_qualification_contract",
        all(qualification_contract.values()),
        qualification_contract,
    )
    publisher = (root / "aegis_research" / "bordair_dual_publish.py").read_text(
        encoding="utf-8"
    )
    publisher_cli = (root / "scripts" / "publish_bordair_dual_pair.py").read_text(
        encoding="utf-8"
    )
    publisher_tests = (root / "tests" / "test_bordair_dual_publish.py").read_text(
        encoding="utf-8"
    )
    publication_contract = {
        "source_and_destination_validation": all(
            token in publisher
            for token in (
                "validate_dual_source_candidate",
                "validate_dual_promotion_candidate",
                "promoted_pair_manifest=pair_manifest",
                "promoted_image_artifact=image",
                "promoted_text_artifact=text",
                "_candidate_runtime_preflight",
            )
        ),
        "atomic_exact_package": all(
            token in publisher
            for token in (
                "qualification_evidence",
                "os.O_EXCL",
                "tempfile.mkdtemp",
                "os.rename(stage, destination)",
                "st_nlink != 1",
                "published_but_quarantined",
                "result.get(\"passed\") is not True",
                "_fsync_directory(destination.parent)",
                "destination exists but is not byte-identical",
                "source changed while publishing",
            )
        ),
        "thin_cli": all(
            token in publisher_cli
            for token in (
                "publish_qualified_dual_pair",
                "--destination",
                "--runtime-config",
            )
        ),
        "entrypoint_corpus_binding": all(
            token in publisher_cli
            for token in (
                "corpus_manifest = run_paths.manifest",
                "manifest_path=corpus_manifest",
                "validate_dual_source_candidate",
                "validate_dual_promotion_candidate",
            )
        ),
        "mutation_tests": all(
            token in publisher_tests
            for token in (
                "test_exact_destination_is_idempotent",
                "test_partial_destination_is_rejected_without_overwrite",
                "test_destination_mutation_is_rejected",
                "test_wrong_target_is_rejected_before_rename",
                "test_crash_before_rename_leaves_no_partial_destination",
                "test_post_validation_failure_quarantines_new_destination",
                "test_false_post_validation_result_is_not_success",
                "test_hardlinked_destination_file_is_rejected",
                "test_extra_empty_directory_is_rejected",
                "test_nested_symlink_is_rejected",
                "test_publisher_entrypoint_validates_against_corpus_manifest",
            )
        ),
        "documentation": all(
            token in readme
            for token in (
                "publish_bordair_dual_pair.py",
                "qualification_evidence/",
                "schema-2 deployment config",
                "partial or divergent contents are never overwritten",
                "bundled Qwen v7 package is published",
                "claim runtime publication only after",
                "exactly 50 frozen",
                "209 files",
                "cache_entry_tree_sha256=f7577b99b3f9e4cfff888fd9fa10f337be9d4fd880eb80a0f73065bc4e5864b7",
                "corpus_manifest_sha256=ffd961e91771ec95b2781a66c8992fd0edafe94dfaf61f5fe4062dde02127220",
                "promotion_authorized_by_evidence: false",
                "not the Hugging Face model cache",
                "$imageId",
                "published_but_quarantined",
                "Remove a stale",
            )
        ),
    }
    _record(
        checks,
        "pipeline_dual_publication_contract",
        all(publication_contract.values()),
        publication_contract,
    )

    qwen_package = REPOSITORY / "models" / "aegis" / "qwen25vl3b_v7_dual_or"
    qwen_manifest_path = qwen_package / "detector_pair_manifest.json"
    qwen_evidence = qwen_package / "qualification_evidence"
    qwen_package_details: dict[str, object] = {}
    qwen_package_ok = False
    try:
        qwen_manifest = _load_json(qwen_manifest_path)
        qwen_qualification = qwen_manifest["qualification_evidence"]
        bound_files = qwen_qualification["files"]
        bound_cache = {
            str(entry["path"]): str(entry["sha256"])
            for name, entry in bound_files.items()
            if str(name).startswith("cache::")
        }
        physical_evidence_files = {
            path.relative_to(qwen_package).as_posix()
            for path in qwen_evidence.rglob("*")
            if path.is_file()
        }
        physical_cache_files = {
            path.relative_to(qwen_package).as_posix()
            for path in (qwen_evidence / "cache").rglob("*")
            if path.is_file()
        }
        bound_cache_hashes_match = all(
            path in physical_cache_files
            and _raw_sha256(qwen_package / path) == expected_sha256
            for path, expected_sha256 in bound_cache.items()
        )
        qwen_package_details = {
            "manifest_sha256": _raw_sha256(qwen_manifest_path),
            "runtime_detector_identity_sha256": qwen_manifest.get(
                "runtime_detector_identity_sha256"
            ),
            "pair_identity_sha256": qwen_manifest.get("pair_identity_sha256"),
            "evidence_file_count": len(physical_evidence_files),
            "bound_file_count": len(bound_files),
            "physical_cache_file_count": len(physical_cache_files),
            "bound_cache_file_count": len(bound_cache),
            "bound_cache_hashes_match": bound_cache_hashes_match,
            "cache_entry_tree_sha256": qwen_qualification.get(
                "cache_entry_tree_sha256"
            ),
            "corpus_manifest_sha256": qwen_qualification.get(
                "corpus_manifest_sha256"
            ),
            "promotion_authorized_by_evidence": qwen_qualification.get(
                "promotion_authorized_by_evidence"
            ),
        }
        qwen_package_ok = (
            qwen_package_details["manifest_sha256"]
            == "3045793da674120f5c46358380278ba65673b8b17e88c86ccb91c9afc59787ca"
            and qwen_package_details["runtime_detector_identity_sha256"]
            == "960f53c6fa3e584bfd2ff16a6ec3cd2d54f592e640e32da55748c2b8facd3a8e"
            and qwen_package_details["pair_identity_sha256"]
            == "7d44bf903ec880082aabc569b3468ef719746f87023422c0c721cf3463ff1174"
            and qwen_package_details["evidence_file_count"] == 209
            and qwen_package_details["bound_file_count"] == 209
            and qwen_package_details["physical_cache_file_count"] == 50
            and qwen_package_details["bound_cache_file_count"] == 50
            and bound_cache_hashes_match
            and set(bound_cache) == physical_cache_files
            and qwen_package_details["cache_entry_tree_sha256"]
            == "f7577b99b3f9e4cfff888fd9fa10f337be9d4fd880eb80a0f73065bc4e5864b7"
            and qwen_package_details["corpus_manifest_sha256"]
            == "ffd961e91771ec95b2781a66c8992fd0edafe94dfaf61f5fe4062dde02127220"
            and qwen_package_details["promotion_authorized_by_evidence"] is False
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        qwen_package_details = {"error": str(exc)}
    _record(
        checks,
        "pipeline_qwen_v7_published_package_binding",
        qwen_package_ok,
        qwen_package_details,
    )

    deliverables_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    _record(
        checks,
        "pipeline_qwen_v7_publication_documented",
        all(
            token in deliverables_readme
            for token in (
                "209 qualification-evidence files",
                "exactly 50 frozen cache entries",
                "cache_entry_tree_sha256=f7577b99b3f9e4cfff888fd9fa10f337be9d4fd880eb80a0f73065bc4e5864b7",
                "corpus_manifest_sha256=ffd961e91771ec95b2781a66c8992fd0edafe94dfaf61f5fe4062dde02127220",
                "promotion_authorized_by_evidence: false",
                "not the Hugging Face model cache",
            )
        ),
    )
    _record(
        checks,
        "pipeline_provenance_documented",
        "provenance" in readme.lower()
        and "llava05b_v7_dual_or/detector_pair_manifest.json" in readme
        and "qwen25vl3b_v7_dual_or/detector_pair_manifest.json" in readme
        and "pooling=text_image_tokens" in readme
        and "`legacy_single`" in readme
        and "qwen v7" in readme.lower(),
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
    runtime_root = ROOT / "runtime_validation"
    expected_files = {
        "README.md",
        HISTORICAL_RUNTIME_RECORD,
        CURRENT_RUNTIME_RECORD,
    }
    actual_files = _relative_file_set(runtime_root)
    _record(
        checks,
        "runtime_validation_layout",
        actual_files == expected_files,
        {
            "missing": sorted(expected_files - actual_files),
            "unexpected": sorted(actual_files - expected_files),
        },
    )
    record_path = runtime_root / HISTORICAL_RUNTIME_RECORD
    record = _load_json(record_path)
    record_sha256 = _raw_sha256(record_path)
    pyproject = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
    package_version = version_match.group(1) if version_match else None
    expected_record_keys = {
        "schema_version",
        "validation_date",
        "status",
        "record_classification",
        "scope",
        "package_version",
        "evidence_boundary",
        "targets",
        "pending_checks",
    }
    record_identity_ok = (
        record_sha256 == HISTORICAL_RUNTIME_RECORD_SHA256
        and set(record) == expected_record_keys
        and record["schema_version"] == 4
        and record["validation_date"] == "2026-08-29"
        and record["status"] == "partial"
        and record["record_classification"] == "current_v6_validation"
        and record["package_version"] == package_version
        and set(record["targets"]) == {"llava05b", "qwen25vl3b"}
        and "Bordair OCR v6 targets" in record["scope"]
        and "not claimed as passed" in record["evidence_boundary"]
    )
    _record(
        checks,
        "runtime_historical_record_identity",
        record_identity_ok,
        {
            "schema_version": record.get("schema_version"),
            "status": record.get("status"),
            "package_version": package_version,
            "targets": sorted(record.get("targets", {})),
            "record_sha256": record_sha256,
            "embedded_classification": record.get("record_classification"),
            "effective_classification": "historical_superseded",
        },
    )

    expected_dimensions = {"llava05b": 896, "qwen25vl3b": 2048}
    artifact_details: dict[str, object] = {}
    artifact_contracts_ok = True
    recorded_paths_ok = True

    for target, expected_sha256 in HISTORICAL_RUNTIME_ARTIFACT_SHA256.items():
        target_record = record["targets"][target]
        artifact_record = target_record["artifact"]
        recorded_paths_ok &= (
            target_record["config"] == HISTORICAL_RUNTIME_CONFIG_PATHS[target]
        )
        artifact_relative = Path(artifact_record["path"])
        artifact_path = (REPOSITORY / artifact_relative).resolve()
        try:
            artifact_path.relative_to(REPOSITORY.resolve())
            artifact_path_safe = not artifact_relative.is_absolute()
        except ValueError:
            artifact_path_safe = False
        if not artifact_path_safe:
            raise ValueError(
                f"{target} detector path must remain inside the repository"
            )
        # The pinned record preserves the v6 validation evidence after its
        # superseded weights have been retired. Current v7 heads are loaded and
        # verified separately below; historical metadata is not a live load.
        target_artifact_ok = (
            record_sha256 == HISTORICAL_RUNTIME_RECORD_SHA256
            and artifact_record["sha256"] == expected_sha256
            and artifact_record["feature_dim"] == expected_dimensions[target]
            and artifact_record["pooling"] == "image_tokens"
            and artifact_record["layer"] == -1
            and all(
                isinstance(artifact_record[key], str) and artifact_record[key]
                for key in (
                    "source",
                    "model_family",
                    "model_id",
                    "model_revision",
                    "tokenizer_revision",
                    "preprocessing_sha256",
                )
            )
            and re.fullmatch(
                r"[0-9a-f]{64}", str(artifact_record["preprocessing_sha256"])
            )
            is not None
            and math.isfinite(float(artifact_record["block_threshold"]))
            and 0.0 <= float(artifact_record["block_threshold"]) <= 1.0
        )
        artifact_contracts_ok &= target_artifact_ok
        artifact_details[target] = {
            "validation_basis": "immutable_historical_record",
            "sha256": artifact_record["sha256"],
            "feature_dim": artifact_record["feature_dim"],
            "pooling": artifact_record["pooling"],
            "layer": artifact_record["layer"],
        }

    _record(
        checks,
        "runtime_v6_historical_recorded_artifact_contracts",
        artifact_contracts_ok,
        artifact_details,
    )
    _record(
        checks,
        "runtime_v6_historical_recorded_paths",
        recorded_paths_ok,
        HISTORICAL_RUNTIME_CONFIG_PATHS,
    )

    qwen_cache = record["targets"]["qwen25vl3b"]["runtime_validation"].get(
        "model_cache_preflight", {}
    )
    qwen_cache_ok = (
        qwen_cache.get("status") == "pass"
        and qwen_cache.get("storage_mode") == "huggingface_cache"
        and qwen_cache.get("snapshot_revision")
        == "66285546d2b821cf421d4f5eb2576359d3770cd3"
        and qwen_cache.get("fingerprint_scheme")
        == "transformers_pytorch_runtime_v1"
        and qwen_cache.get("fingerprint_sha256")
        == "45f7d1afd0ef8e09cb7a79456fd232014d2a482ca975d2008848c0efa77e4ce0"
        and qwen_cache.get("file_count") == 11
        and qwen_cache.get("total_bytes") == 7_520_892_432
        and qwen_cache.get("download_performed") is False
        and qwen_cache.get("prepare_ready") is True
    )
    _record(checks, "runtime_v6_qwen_cache_preflight", qwen_cache_ok)

    required_live_checks = {
        "compose_build",
        "cuda_model_load",
        "warmup",
        "authenticated_image_text_inference",
    }
    live_values = {
        target: {
            name: record["targets"][target]["runtime_validation"].get(name)
            for name in required_live_checks
        }
        for target in ("llava05b", "qwen25vl3b")
    }

    def positive_number(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
        )

    def target_live_evidence_passes(target: str) -> bool:
        runtime = record["targets"][target]["runtime_validation"]
        doctor = runtime.get("doctor", {})
        startup = runtime.get("startup", {})
        inference = runtime.get("inference", {})
        risk_score = inference.get("risk_score")
        return (
            all(value == "pass" for value in live_values[target].values())
            and doctor.get("status") == "pass"
            and positive_number(doctor.get("required_checks_total"))
            and doctor.get("required_checks_passed")
            == doctor.get("required_checks_total")
            and doctor.get("detector_sha256_match") is True
            and doctor.get("provider_options_exact_match") is True
            and doctor.get("model_and_tokenizer_revisions_exact_match") is True
            and doctor.get("model_content_sha256_match") is True
            and positive_number(doctor.get("cuda_free_memory_bytes"))
            and runtime.get("compose_image", {}).get("image_sha256")
            == "f4faef3c4e66d8470802728197fd3074577cd5221290660aa645273f9d4c7a6d"
            and positive_number(startup.get("ready_seconds"))
            and startup.get("worker_mode") == "process"
            and positive_number(startup.get("worker_generation"))
            and startup.get("worker_restarts") == 0
            and re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                str(startup.get("evidence_session_id", "")),
            )
            is not None
            and inference.get("http_status") == 200
            and positive_number(inference.get("elapsed_seconds"))
            and inference.get("modality") == "image_text"
            and inference.get("detector_source")
            == record["targets"][target]["artifact"]["source"]
            and inference.get("verdict") in {"benign", "malicious"}
            and inference.get("action") == "allow"
            and isinstance(risk_score, (int, float))
            and not isinstance(risk_score, bool)
            and math.isfinite(float(risk_score))
            and 0.0 <= float(risk_score) <= 1.0
        )

    if record["status"] == "pending":
        live_status_ok = (
            all(
                value == "pending"
                for values in live_values.values()
                for value in values.values()
            )
            and len(record["pending_checks"]) == 6
        )
    elif record["status"] == "pass":
        live_status_ok = (
            all(
                target_live_evidence_passes(target)
                for target in ("llava05b", "qwen25vl3b")
            )
            and record["pending_checks"] == []
        )
    else:
        llava_runtime = record["targets"]["llava05b"]["runtime_validation"]
        qwen_runtime = record["targets"]["qwen25vl3b"]["runtime_validation"]
        llava_doctor = llava_runtime.get("doctor", {})
        llava_inference = llava_runtime.get("inference", {})
        qwen_resources = qwen_runtime.get("resource_preflight", {})
        qwen_required = HISTORICAL_QWEN_RESOURCE_REQUIREMENTS
        expected_pending = {
            "qwen25vl3b.cuda_model_load",
            "qwen25vl3b.warmup",
            "qwen25vl3b.authenticated_image_text_inference",
        }
        live_status_ok = (
            record["status"] == "partial"
            and target_live_evidence_passes("llava05b")
            and live_values["qwen25vl3b"]["compose_build"] == "pass"
            and all(
                live_values["qwen25vl3b"][name]
                == "not_run_resource_preflight"
                for name in (
                    "cuda_model_load",
                    "warmup",
                    "authenticated_image_text_inference",
                )
            )
            and qwen_resources.get("status") == "fail_closed"
            and qwen_resources.get("container_exit_code") == 1
            and set(qwen_resources.get("failed_checks", []))
            == {"total_physical_memory", "cuda_device_available_memory"}
            and qwen_resources.get("total_physical_bytes", 0)
            < qwen_resources.get("required_total_physical_bytes", 0)
            and qwen_resources.get("required_total_physical_bytes")
            == qwen_required["min_total_physical_bytes"]
            and qwen_resources.get("available_physical_bytes", 0)
            >= qwen_resources.get("required_available_physical_bytes", 0)
            and qwen_resources.get("required_available_physical_bytes")
            == qwen_required["min_available_physical_bytes"]
            and qwen_resources.get("available_virtual_bytes", 0)
            >= qwen_resources.get("required_available_virtual_bytes", 0)
            and qwen_resources.get("required_available_virtual_bytes")
            == qwen_required["min_available_virtual_bytes"]
            and qwen_resources.get("cuda_free_memory_bytes", 0)
            < qwen_resources.get("required_cuda_free_memory_bytes", 0)
            and qwen_resources.get("required_cuda_free_memory_bytes")
            == qwen_required["min_cuda_device_memory_bytes"]
            and qwen_resources.get("model_cache_bytes", 0)
            >= qwen_required["min_model_cache_bytes"]
            and set(record["pending_checks"]) == expected_pending
            and len(record["pending_checks"]) == len(expected_pending)
        )
    _record(
        checks,
        "runtime_v6_historical_live_evidence_status_explicit",
        live_status_ok,
        {"status": record["status"], "checks": live_values},
    )

    current_path = runtime_root / CURRENT_RUNTIME_RECORD
    current = _load_json(current_path)
    current_sha256 = _raw_sha256(current_path)
    current_expected_keys = {
        "schema_version",
        "validation_date",
        "status",
        "record_classification",
        "scope",
        "package_version",
        "evidence_boundary",
        "secrets_omitted",
        "targets",
        "pending_checks",
    }
    current_pending = {
        "qwen25vl3b.cuda_model_load",
        "qwen25vl3b.warmup",
        "qwen25vl3b.authenticated_image_text_inference",
    }
    current_identity_ok = (
        current_sha256 == CURRENT_RUNTIME_RECORD_SHA256
        and set(current) == current_expected_keys
        and current.get("schema_version") == 5
        and current.get("validation_date") == "2026-08-30"
        and current.get("status") == "partial"
        and current.get("record_classification") == "current_v7_validation"
        and current.get("package_version") == package_version
        and current.get("secrets_omitted") is True
        and set(current.get("targets", {})) == {"llava05b", "qwen25vl3b"}
        and current.get("scope")
        == (
            "Rebuilt the current AEGIS 0.3.0 Compose services with the v7 dual-head "
            "OR detector packages, completed CUDA model loading and authenticated "
            "image-text inference for LLaVA, exercised the local dashboard, and "
            "attempted Qwen through its unchanged resource preflight."
        )
        and current.get("evidence_boundary")
        == (
            "LLaVA v7 completed startup, authenticated status, warmup, two fixed-panel "
            "dashboard inferences, and graceful shutdown. Qwen v7 was built and its "
            "detector package was present, but the service failed closed before model "
            "loading because the Docker VM did not meet the configured total-RAM and "
            "free-VRAM minima. Qwen warmup and live inference are not claimed as passed."
        )
        and set(current.get("pending_checks", [])) == current_pending
        and len(current.get("pending_checks", [])) == len(current_pending)
    )
    _record(
        checks,
        "runtime_v7_current_record_identity",
        current_identity_ok,
        {
            "schema_version": current.get("schema_version"),
            "status": current.get("status"),
            "record_sha256": current_sha256,
            "classification": current.get("record_classification"),
            "pending_checks": current.get("pending_checks"),
        },
    )

    current_head_contracts = {
        "llava05b": {
            "config": "configs/aegis.llava.deployment.container.json",
            "feature_dim": 1792,
            "heads": {
                "text": {
                    "path": "models/aegis/llava05b_v7_dual_or/llava05b_text_head_v1.npz",
                    "sha256": "cd8502fc73ecaf1e6597d318d63a82fcdf7abae628c1a54e5390b82b2a999a61",
                    "pooling": "text_tokens",
                    "feature_dim": 896,
                    "review_threshold": 0.9773003604375604,
                    "block_threshold": 0.9973003604375604,
                },
                "image": {
                    "path": "models/aegis/llava05b_v7_dual_or/llava05b_image_head_v1.npz",
                    "sha256": "84b12df0887f420318e2f3f530d0dc4391e8c1e1e5d1e15ff28f4f719c6b80b5",
                    "pooling": "image_tokens",
                    "feature_dim": 896,
                    "review_threshold": 0.41109073768976995,
                    "block_threshold": 0.43109073768976996,
                },
            },
        },
        "qwen25vl3b": {
            "config": "configs/aegis.deployment.container.json",
            "feature_dim": 4096,
            "heads": {
                "text": {
                    "path": "models/aegis/qwen25vl3b_v7_dual_or/qwen25vl3b_text_head_v1.npz",
                    "sha256": "8818a75eb0fbe9fb1a0acb9f2035cb8b9f48f083ea1ee7aee03e15ec549533c8",
                    "pooling": "text_tokens",
                    "feature_dim": 2048,
                    "review_threshold": 0.9775257227486033,
                    "block_threshold": 0.9975257227486033,
                },
                "image": {
                    "path": "models/aegis/qwen25vl3b_v7_dual_or/qwen25vl3b_image_head_v1.npz",
                    "sha256": "e17eff8de97b45c22887139417d84bdd32ed067c9615c0d601f9af4467b18deb",
                    "pooling": "image_tokens",
                    "feature_dim": 2048,
                    "review_threshold": 0.1490249723273814,
                    "block_threshold": 0.1690249723273814,
                },
            },
        },
    }
    current_artifact_details: dict[str, object] = {}
    current_artifact_contracts_ok = True
    current_configs: dict[str, dict[str, object]] = {}
    for target, expected_target in current_head_contracts.items():
        target_record = current["targets"][target]
        detector_record = target_record["detector"]
        config_path = REPOSITORY / str(expected_target["config"])
        config = _load_json(config_path)
        current_configs[target] = config
        config_heads = {
            str(head["name"]): head for head in config["detector"]["heads"]
        }
        record_heads = {
            str(head["name"]): head for head in detector_record["heads"]
        }
        expected_heads = expected_target["heads"]
        target_ok = (
            target_record.get("config") == expected_target["config"]
            and config.get("schema_version") == 2
            and config.get("target_profile") == target
            and config.get("traffic_mode") == "shadow"
            and config.get("detector", {}).get("mode") == "or"
            and config.get("provider_options", {}).get("pooling")
            == "text_image_tokens"
            and config.get("provider_options", {}).get("feature_dim")
            == expected_target["feature_dim"]
            and detector_record.get("mode") == "or"
            and detector_record.get("identity_sha256")
            == CURRENT_RUNTIME_IDENTITIES[target]
            and detector_record.get("feature_dim") == expected_target["feature_dim"]
            and set(config_heads) == set(expected_heads)
            and set(record_heads) == set(expected_heads)
        )
        if target == "qwen25vl3b":
            target_ok &= (
                detector_record.get("published_pair_identity_sha256")
                == CURRENT_QWEN_PAIR_IDENTITY
            )
        else:
            target_ok &= "published_pair_identity_sha256" not in detector_record

        head_details: dict[str, object] = {}
        for head_name, expected_head in expected_heads.items():
            head_record = record_heads.get(head_name, {})
            config_head = config_heads.get(head_name, {})
            artifact_relative = Path(str(expected_head["path"]))
            artifact_path = (REPOSITORY / artifact_relative).resolve()
            try:
                artifact_path.relative_to(REPOSITORY.resolve())
                artifact_safe = not artifact_relative.is_absolute()
            except ValueError:
                artifact_safe = False
            if not artifact_safe:
                raise ValueError(
                    f"{target}/{head_name} detector path must remain inside the repository"
                )
            artifact_sha256 = _raw_sha256(artifact_path)
            with np.load(artifact_path, allow_pickle=False) as artifact:
                artifact_dim = int(np.asarray(artifact["weights"]).size)
                artifact_pooling = str(
                    np.asarray(artifact["pooling"]).reshape(-1)[0]
                )
                artifact_block = float(
                    np.asarray(artifact["threshold"]).reshape(-1)[0]
                )
            head_ok = (
                head_record.get("path") == expected_head["path"]
                and config_head.get("artifact") == expected_head["path"]
                and artifact_sha256
                == head_record.get("sha256")
                == expected_head["sha256"]
                and head_record.get("pooling")
                == artifact_pooling
                == expected_head["pooling"]
                and head_record.get("feature_dim")
                == artifact_dim
                == expected_head["feature_dim"]
                and math.isclose(
                    float(head_record.get("review_threshold")),
                    float(config_head.get("review_threshold")),
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
                and math.isclose(
                    float(head_record.get("review_threshold")),
                    float(expected_head["review_threshold"]),
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
                and math.isclose(
                    float(head_record.get("block_threshold")),
                    artifact_block,
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
                and math.isclose(
                    float(head_record.get("block_threshold")),
                    float(expected_head["block_threshold"]),
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
            )
            target_ok &= head_ok
            head_details[head_name] = {
                "sha256": artifact_sha256,
                "pooling": artifact_pooling,
                "feature_dim": artifact_dim,
                "passed": head_ok,
            }
        current_artifact_contracts_ok &= target_ok
        current_artifact_details[target] = {
            "identity_sha256": detector_record.get("identity_sha256"),
            "feature_dim": detector_record.get("feature_dim"),
            "heads": head_details,
            "passed": target_ok,
        }
    _record(
        checks,
        "runtime_v7_current_artifact_contracts",
        current_artifact_contracts_ok,
        current_artifact_details,
    )

    llava_current = current["targets"]["llava05b"]["runtime_validation"]
    llava_launcher = llava_current.get("windows_launcher", {})
    llava_ready = llava_current.get("readiness", {})
    llava_status = llava_current.get("authenticated_status", {})
    llava_warmup = llava_current.get("authenticated_warmup_inference", {})
    llava_fixed = llava_current.get("fixed_panel_live_inference", {})
    llava_cases = {
        str(case.get("sample_id")): case for case in llava_fixed.get("cases", [])
    }
    llava_benign = llava_cases.get("BTI-05697", {})
    llava_malicious = llava_cases.get("TI-01093", {})
    llava_dashboard = llava_current.get("dashboard", {})
    llava_post_doctor = llava_current.get("post_load_doctor", {})
    llava_shutdown = llava_current.get("shutdown", {})
    llava_current_ok = (
        llava_current.get("compose_build") == "pass"
        and llava_current.get("compose_image_id")
        == "sha256:ae337c6da1a206d473d412de7642ff9adbabd524e845b32f6e0f62b748928341"
        and llava_launcher.get("status") == "pass"
        and llava_launcher.get("target") == "llava05b"
        and llava_launcher.get("count_property_failure_reproduced") is False
        and llava_launcher.get("authenticated_detector_identity_match") is True
        and llava_current.get("startup_resource_preflight") == "pass"
        and llava_current.get("cuda_model_load") == "pass"
        and llava_current.get("warmup") == "pass"
        and llava_ready.get("http_status") == 200
        and llava_ready.get("ok") is True
        and llava_ready.get("target_profile") == "llava05b"
        and llava_ready.get("traffic_mode") == "shadow"
        and llava_ready.get("input_modalities") == ["image_text"]
        and llava_status.get("http_status") == 200
        and llava_status.get("detector_mode") == "or"
        and llava_status.get("detector_identity_sha256")
        == CURRENT_RUNTIME_IDENTITIES["llava05b"]
        and llava_status.get("head_count") == 2
        and llava_status.get("worker_mode") == "process"
        and llava_status.get("worker_generation") == 1
        and llava_status.get("worker_restarts") == 0
        and llava_warmup.get("status") == "pass"
        and llava_warmup.get("request_id") == "llava-warmup-benign-001"
        and llava_warmup.get("verdict") == "benign"
        and llava_warmup.get("recommended_action") == "allow"
        and llava_warmup.get("effective_action") == "allow"
        and set(llava_cases) == {"BTI-05697", "TI-01093"}
        and len(llava_fixed.get("cases", [])) == 2
        and llava_fixed.get("status") == "pass"
        and llava_fixed.get("traffic_mode") == "shadow"
        and llava_benign.get("expected_label") == "benign"
        and llava_benign.get("verdict") == "benign"
        and llava_benign.get("recommended_action") == "allow"
        and llava_benign.get("effective_action") == "allow"
        and llava_malicious.get("expected_label") == "malicious"
        and llava_malicious.get("verdict") == "malicious"
        and llava_malicious.get("recommended_action") == "block"
        and llava_malicious.get("effective_action") == "allow"
        and llava_malicious.get("effective_action_reason") == "shadow_mode"
        and all(
            positive_number(case.get("duration_ms"))
            and isinstance(case.get("risk_score"), (int, float))
            and not isinstance(case.get("risk_score"), bool)
            and 0.0 <= float(case["risk_score"]) <= 1.0
            and set(case.get("head_scores", {})) == {"text", "image"}
            for case in llava_cases.values()
        )
        and llava_dashboard.get("status") == "pass"
        and llava_dashboard.get("loopback_origin") == "http://127.0.0.1:8767/"
        and llava_dashboard.get("session_secret_persisted_in_record") is False
        and len(llava_dashboard.get("screenshots", [])) == 5
        and llava_post_doctor.get("status") == "partial"
        and llava_post_doctor.get("required_checks_passed") == 38
        and llava_post_doctor.get("required_checks_total") == 39
        and llava_post_doctor.get("failed_check")
        == "resource_preflight.available_virtual_memory"
        and llava_post_doctor.get("observed_available_virtual_bytes")
        == 4_216_020_992
        and llava_post_doctor.get("required_available_virtual_bytes")
        == current_configs["llava05b"]["resources"]["min_available_virtual_bytes"]
        == 4_294_967_296
        and llava_post_doctor.get("observed_available_virtual_bytes")
        < llava_post_doctor.get("required_available_virtual_bytes")
        and "after the model was already loaded and serving"
        in str(llava_post_doctor.get("context"))
        and "no requirement was reduced" in str(llava_post_doctor.get("context"))
        and llava_shutdown.get("status") == "pass"
        and llava_shutdown.get("container_exit_code") == 0
        and llava_shutdown.get("oom_killed") is False
        and llava_shutdown.get("dashboard_graceful_shutdown") is True
    )
    _record(
        checks,
        "runtime_v7_llava_functional_evidence",
        llava_current_ok,
        {
            "identity_sha256": llava_status.get("detector_identity_sha256"),
            "fixed_panel_cases": sorted(llava_cases),
            "post_load_doctor": llava_post_doctor,
            "shutdown": llava_shutdown,
        },
    )

    qwen_current = current["targets"]["qwen25vl3b"]["runtime_validation"]
    qwen_image = qwen_current.get("compose_image", {})
    qwen_resources = qwen_current.get("resource_preflight", {})
    qwen_config_resources = current_configs["qwen25vl3b"]["resources"]
    qwen_shutdown = qwen_current.get("shutdown", {})
    qwen_current_ok = (
        qwen_current.get("compose_build") == "pass"
        and qwen_image.get("repository") == "aegis-mllm-guard:local"
        and qwen_image.get("image_id")
        == "sha256:2d2a6d5208b4b0e2e07ae3b1def65880bdb95915157841870d8d5c59dc6b6e30"
        and positive_number(qwen_image.get("size_bytes"))
        and qwen_image.get("architecture") == "amd64"
        and qwen_image.get("os") == "linux"
        and qwen_current.get("detector_package_present") is True
        and qwen_resources.get("status") == "fail_closed"
        and qwen_resources.get("container_exit_code") == 1
        and qwen_resources.get("oom_killed") is False
        and set(qwen_resources.get("failed_checks", []))
        == {"total_physical_memory", "cuda_device_available_memory"}
        and qwen_resources.get("total_physical_bytes") == 8_172_244_992
        and qwen_resources.get("required_total_physical_bytes")
        == qwen_config_resources["min_total_physical_bytes"]
        == 16_106_127_360
        and qwen_resources.get("total_physical_bytes")
        < qwen_resources.get("required_total_physical_bytes")
        and qwen_resources.get("available_physical_bytes") == 6_859_546_624
        and qwen_resources.get("required_available_physical_bytes")
        == qwen_config_resources["min_available_physical_bytes"]
        == 6_442_450_944
        and qwen_resources.get("available_physical_bytes")
        >= qwen_resources.get("required_available_physical_bytes")
        and qwen_resources.get("available_virtual_bytes") == 8_789_344_256
        and qwen_resources.get("required_available_virtual_bytes")
        == qwen_config_resources["min_available_virtual_bytes"]
        == 8_589_934_592
        and qwen_resources.get("available_virtual_bytes")
        >= qwen_resources.get("required_available_virtual_bytes")
        and qwen_resources.get("disk_free_bytes") == 982_931_873_792
        and qwen_resources.get("required_disk_free_bytes")
        == qwen_config_resources["min_disk_free_bytes"]
        == 12_884_901_888
        and qwen_resources.get("disk_free_bytes")
        >= qwen_resources.get("required_disk_free_bytes")
        and qwen_resources.get("model_cache_bytes") == 15_041_784_864
        and qwen_resources.get("required_model_cache_bytes")
        == qwen_config_resources["min_model_cache_bytes"]
        == 6_442_450_944
        and qwen_resources.get("model_cache_bytes")
        >= qwen_resources.get("required_model_cache_bytes")
        and qwen_resources.get("cuda_device")
        == "NVIDIA GeForce RTX 4060 Laptop GPU"
        and qwen_resources.get("cuda_total_memory_bytes") == 8_585_216_000
        and qwen_resources.get("cuda_free_memory_bytes") == 7_440_695_296
        and qwen_resources.get("required_cuda_free_memory_bytes")
        == qwen_config_resources["min_cuda_device_memory_bytes"]
        == 7_516_192_768
        and qwen_resources.get("cuda_free_memory_bytes")
        < qwen_resources.get("required_cuda_free_memory_bytes")
        and all(
            qwen_current.get(name) == "not_run_resource_preflight"
            for name in (
                "cuda_model_load",
                "warmup",
                "authenticated_image_text_inference",
            )
        )
        and qwen_shutdown.get("status") == "pass_after_fail_closed_startup"
        and qwen_shutdown.get("container_exit_code") == 1
        and qwen_shutdown.get("oom_killed") is False
        and set(current.get("pending_checks", [])) == current_pending
    )
    _record(
        checks,
        "runtime_v7_qwen_fail_closed_boundary",
        qwen_current_ok,
        {
            "configured_detector_identity_sha256": current["targets"]["qwen25vl3b"][
                "detector"
            ].get("identity_sha256"),
            "failed_checks": qwen_resources.get("failed_checks"),
            "total_physical_bytes": qwen_resources.get("total_physical_bytes"),
            "cuda_free_memory_bytes": qwen_resources.get("cuda_free_memory_bytes"),
            "pending_checks": current.get("pending_checks"),
        },
    )

    runtime_readme = (
        ROOT / "runtime_validation" / "README.md"
    ).read_text(encoding="utf-8")
    deliverables_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    repository_readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    repository_readme_normalized = " ".join(repository_readme.lower().split())
    _record(
        checks,
        "runtime_v6_historical_scope_documented",
        "not_run_resource_preflight" in runtime_readme
        and "failed closed" in runtime_readme.lower()
        and "historical" in runtime_readme.lower()
        and "superseded" in runtime_readme.lower()
        and "does not verify either v7 runtime" in runtime_readme.lower()
        and "the json itself remains unchanged" in runtime_readme.lower()
        and "aegis_llava_onevision_05b_bordair_ocr_v6.npz" in runtime_readme
        and HISTORICAL_RUNTIME_RECORD in runtime_readme,
    )
    _record(
        checks,
        "runtime_v7_current_scope_documented",
        CURRENT_RUNTIME_RECORD in runtime_readme
        and CURRENT_RUNTIME_RECORD in deliverables_readme
        and "current v7 contract on 2026-08-30" in runtime_readme.lower()
        and CURRENT_RUNTIME_RECORD_SHA256 in runtime_readme
        and CURRENT_RUNTIME_IDENTITIES["llava05b"] in runtime_readme
        and CURRENT_RUNTIME_IDENTITIES["llava05b"] in deliverables_readme
        and CURRENT_RUNTIME_IDENTITIES["llava05b"] in repository_readme
        and "BTI-05697" in runtime_readme
        and "TI-01093" in runtime_readme
        and "shadow" in runtime_readme.lower()
        and "38 of 39" in runtime_readme
        and "4,216,020,992" in runtime_readme
        and "4,294,967,296" in runtime_readme
        and "8,172,244,992" in runtime_readme
        and "16,106,127,360" in runtime_readme
        and "7,440,695,296" in runtime_readme
        and "7,516,192,768" in runtime_readme
        and "pending_checks" in runtime_readme
        and "not an accuracy" in runtime_readme.lower()
        and "robustness" in runtime_readme.lower()
        and "latency" in runtime_readme.lower()
        and "production-capacity" in runtime_readme.lower()
        and "current 2026-08-30 compose validation" in repository_readme_normalized
        and "its cuda loading, warmup, and live inference remain pending"
        in repository_readme_normalized
        and "8,172,244,992" in deliverables_readme
        and "7,440,695,296" in deliverables_readme
        and "qwen load, warmup, and inference remain pending"
        in deliverables_readme.lower()
        and "llava05b_text_head_v1.npz" in deliverables_readme
        and "llava05b_image_head_v1.npz" in deliverables_readme
        and "qwen25vl3b_text_head_v1.npz" in deliverables_readme
        and "qwen25vl3b_image_head_v1.npz" in deliverables_readme
        and "qwen v7" in deliverables_readme.lower(),
    )
    scope_checks = {
        check["name"]: bool(check["passed"])
        for check in checks
        if check["name"]
        in {
            "runtime_v6_historical_scope_documented",
            "runtime_v7_current_scope_documented",
        }
    }
    _record(
        checks,
        "runtime_evidence_scope_documented",
        len(scope_checks) == 2 and all(scope_checks.values()),
        scope_checks,
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
    historical_runtime_path = ROOT / "runtime_validation" / HISTORICAL_RUNTIME_RECORD
    historical_runtime = _load_json(historical_runtime_path)
    current_runtime_path = ROOT / "runtime_validation" / CURRENT_RUNTIME_RECORD
    current_runtime = _load_json(current_runtime_path)
    report_ok = (
        report.get("schema_version") == REPORT_SCHEMA_VERSION
        and report.get("status") == "pass"
        and report.get("scope") == "current_aegis_deliverables"
        and report.get("package_version")
        == historical_runtime["package_version"]
        == current_runtime["package_version"]
        and report.get("historical_runtime_validation_date")
        == historical_runtime["validation_date"]
        and report.get("historical_runtime_validation_status")
        == historical_runtime["status"]
        and report.get("historical_runtime_record_classification")
        == "historical_superseded"
        and report.get("historical_runtime_record_sha256")
        == HISTORICAL_RUNTIME_RECORD_SHA256
        == _raw_sha256(historical_runtime_path)
        and report.get("current_runtime_validation_date")
        == current_runtime["validation_date"]
        and report.get("current_runtime_validation_status")
        == current_runtime["status"]
        and report.get("current_runtime_record_classification")
        == current_runtime["record_classification"]
        == "current_v7_validation"
        and report.get("current_runtime_record_sha256")
        == CURRENT_RUNTIME_RECORD_SHA256
        == _raw_sha256(current_runtime_path)
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
    historical_runtime = _load_json(
        ROOT / "runtime_validation" / HISTORICAL_RUNTIME_RECORD
    )
    current_runtime = _load_json(ROOT / "runtime_validation" / CURRENT_RUNTIME_RECORD)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "pass" if _all_passed(checks) else "fail",
        "scope": "current_aegis_deliverables",
        "package_version": current_runtime["package_version"],
        "historical_runtime_validation_date": historical_runtime["validation_date"],
        "historical_runtime_validation_status": historical_runtime["status"],
        "historical_runtime_record_classification": "historical_superseded",
        "historical_runtime_record_sha256": HISTORICAL_RUNTIME_RECORD_SHA256,
        "current_runtime_validation_date": current_runtime["validation_date"],
        "current_runtime_validation_status": current_runtime["status"],
        "current_runtime_record_classification": current_runtime[
            "record_classification"
        ],
        "current_runtime_record_sha256": CURRENT_RUNTIME_RECORD_SHA256,
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
