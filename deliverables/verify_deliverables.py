from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REQUIRED_DIRECTORIES = {
    "reproducible_pipeline",
    "red_teaming",
    "annotated_benchmark",
    "ablation_report",
}


def main() -> None:
    checks: list[dict[str, object]] = []
    _record(checks, "deliverable_subdirectories", REQUIRED_DIRECTORIES <= {path.name for path in ROOT.iterdir() if path.is_dir()})
    benchmark = _verify_benchmark(checks)
    _verify_redteam(checks)
    _verify_pipeline(checks, benchmark)
    _verify_ablation(checks)
    _verify_python_syntax(checks)
    manifest = _build_manifest()
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "status": "pass" if passed else "fail",
        "checks_passed": sum(bool(check["passed"]) for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "manifest_files": manifest["file_count"],
        "manifest_total_bytes": manifest["total_bytes"],
    }
    (ROOT / "verification_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


def _verify_benchmark(checks: list[dict[str, object]]) -> list[dict[str, str]]:
    root = ROOT / "annotated_benchmark"
    csv_path = root / "data" / "benchmark.csv"
    rows = _read_csv(csv_path)
    _record(checks, "benchmark_row_count_64", len(rows) == 64, len(rows))
    _record(checks, "benchmark_balanced", Counter(row["label"] for row in rows) == Counter({"benign": 32, "malicious": 32}))
    _record(checks, "benchmark_split_balance", Counter((row["split"], row["label"]) for row in rows) == Counter({("train", "benign"): 16, ("train", "malicious"): 16, ("validation", "benign"): 8, ("validation", "malicious"): 8, ("test", "benign"): 8, ("test", "malicious"): 8}))
    _record(checks, "benchmark_unique_ids", len({row["sample_id"] for row in rows}) == len(rows))
    _record(checks, "benchmark_unique_prompt_hashes", len({row["prompt_sha256"] for row in rows}) == len(rows))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    groups_ok = len(groups) == 32 and all(
        len(group) == 2
        and {row["label_id"] for row in group} == {"0", "1"}
        and len({row["split"] for row in group}) == 1
        for group in groups.values()
    )
    _record(checks, "benchmark_pair_group_isolation", groups_ok, len(groups))
    attack_counts = Counter(row["attack_style"] for row in rows if row["label_id"] == "1")
    _record(checks, "benchmark_eight_attack_styles", len(attack_counts) == 8 and set(attack_counts.values()) == {4}, dict(attack_counts))
    _record(checks, "benchmark_modalities", Counter(row["modality"] for row in rows) == Counter({"image_text": 44, "text": 20}))
    hash_ok = True
    image_count = 0
    for row in rows:
        normalized = " ".join(row["prompt_text"].lower().split())
        hash_ok &= hashlib.sha256(normalized.encode("utf-8")).hexdigest() == row["prompt_sha256"]
        if row["image_path"]:
            image_count += 1
            image_path = root / row["image_path"]
            hash_ok &= image_path.is_file() and _sha256(image_path) == row["image_sha256"]
    _record(checks, "benchmark_content_hashes", hash_ok and image_count == 44, image_count)
    json_rows = json.loads((root / "data" / "benchmark_rows.json").read_text(encoding="utf-8"))
    _record(checks, "benchmark_csv_json_alignment", [row["sample_id"] for row in rows] == [row["sample_id"] for row in json_rows])
    _record(checks, "benchmark_workbook_exists", (root / "benchmark.xlsx").stat().st_size > 10000)
    workbook_qa = (root / "qa" / "workbook_inspect.ndjson").read_text(encoding="utf-8")
    _record(checks, "benchmark_workbook_formula_scan", "Cell search matched 0 entries" in workbook_qa and '"Balanced?","PASS"' in workbook_qa)
    return rows


def _verify_redteam(checks: list[dict[str, object]]) -> None:
    root = ROOT / "red_teaming" / "generated"
    rows = _read_csv(root / "variants.csv")
    _record(checks, "redteam_variant_rows_48", len(rows) == 48, len(rows))
    groups = Counter(row["base_sample_id"] for row in rows)
    _record(checks, "redteam_six_variants_per_sample", len(groups) == 8 and set(groups.values()) == {6}, dict(groups))
    _record(checks, "redteam_variant_families", len({row["variant_type"] for row in rows}) == 6)
    hash_ok = True
    for row in rows:
        normalized = " ".join(row["prompt_text"].lower().split())
        hash_ok &= hashlib.sha256(normalized.encode("utf-8")).hexdigest() == row["prompt_sha256"]
        if row["image_path"]:
            path = root / row["image_path"]
            hash_ok &= path.is_file() and _sha256(path) == row["image_sha256"]
    _record(checks, "redteam_content_hashes", hash_ok)
    scored = _read_csv(root / "scored_variants.csv")
    _record(checks, "redteam_scored_alignment", len(scored) == 48 and all(0.0 <= float(row["risk_score"]) <= 1.0 for row in scored))
    adaptive = json.loads((root / "adaptive_summary.json").read_text(encoding="utf-8"))["summary"]
    evasions = [float(row["evasion_rate"]) for row in adaptive]
    _record(checks, "redteam_bounded_adaptive_budgets", [int(row["query_budget"]) for row in adaptive] == [1, 3, 6])
    _record(checks, "redteam_adaptive_monotonic", evasions == sorted(evasions), evasions)
    _record(checks, "redteam_attack_success_evaluator", (ROOT / "red_teaming" / "evaluate_attack_success.py").is_file())


def _verify_pipeline(checks: list[dict[str, object]], benchmark: list[dict[str, str]]) -> None:
    root = ROOT / "reproducible_pipeline"
    bundle_path = root / "fixtures" / "smoke_features.npz"
    with np.load(bundle_path, allow_pickle=False) as bundle:
        ids = np.asarray(bundle["sample_ids"]).astype(str)
        text = np.asarray(bundle["text_embeddings"])
        image = np.asarray(bundle["image_embeddings"])
        attribution = np.asarray(bundle["attribution_features"])
        source = str(np.asarray(bundle["feature_source"]).reshape(-1)[0])
    _record(checks, "pipeline_feature_alignment", np.array_equal(ids, np.asarray([row["sample_id"] for row in benchmark])))
    _record(checks, "pipeline_feature_shapes", text.shape == (64, 32) and image.shape == (64, 32) and attribution.shape == (64, 8))
    _record(checks, "pipeline_smoke_label", source == "deterministic_smoke_fixture_not_mllm_evidence")
    artifact_root = root / "artifacts" / "smoke"
    _record(checks, "pipeline_detector_artifact", (artifact_root / "selected_detector.npz").stat().st_size > 1000)
    summary = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
    _record(checks, "pipeline_validation_only_selection_documented", summary["selection_rule"].startswith("maximum validation"))
    _record(checks, "pipeline_mllm_extractor", (root / "scripts" / "extract_mllm_features.py").is_file())


def _verify_ablation(checks: list[dict[str, object]]) -> None:
    root = ROOT / "ablation_report"
    results = root / "results"
    feature_rows = _read_csv(results / "feature_signal_ablation.csv")
    low_runs = _read_csv(results / "low_label_seed_runs.csv")
    transfers = _read_csv(results / "attack_family_transfer.csv")
    uncertainty = _read_csv(results / "uncertainty_coverage.csv")
    adaptive = _read_csv(results / "adaptive_redteam_summary.csv")
    _record(checks, "ablation_seven_signal_views", len(feature_rows) == 7)
    _record(checks, "ablation_low_label_seed_sweep", len(low_runs) == 20)
    _record(checks, "ablation_sixteen_transfer_rows", len(transfers) == 16)
    _record(checks, "ablation_uncertainty_coverage", len(uncertainty) == 4)
    _record(checks, "ablation_adaptive_results", len(adaptive) == 3)
    report = (root / "REPORT.md").read_text(encoding="utf-8")
    required_sections = (
        "## Evidence status",
        "## Signal importance",
        "## Low-label and pseudo-label ablation",
        "## Attack-family transfer",
        "## Uncertainty and review coverage",
        "## Bounded adaptive red-team evaluation",
        "## Legacy MLLM evidence",
    )
    _record(checks, "ablation_report_sections", all(section in report for section in required_sections))
    _record(checks, "ablation_smoke_disclaimer", "not evidence of MLLM safety performance" in report)


def _verify_python_syntax(checks: list[dict[str, object]]) -> None:
    files = [path for path in _iter_files() if path.suffix == ".py"]
    failures = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}:{exc.lineno}")
    _record(checks, "python_syntax", not failures, failures or len(files))


def _build_manifest() -> dict[str, object]:
    excluded = {"MANIFEST.json", "verification_report.json"}
    entries = []
    for path in sorted(_iter_files()):
        if path.name in excluded:
            continue
        entries.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {
        "schema_version": 1,
        "root": "deliverables",
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "files": entries,
    }


def _iter_files():
    for directory, names, files in os.walk(ROOT, followlinks=False):
        names[:] = [name for name in names if name not in {"node_modules", "__pycache__"}]
        for name in files:
            yield Path(directory) / name


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _record(checks: list[dict[str, object]], name: str, passed: bool, detail: object | None = None) -> None:
    item: dict[str, object] = {"name": name, "passed": bool(passed)}
    if detail is not None:
        item["detail"] = detail
    checks.append(item)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
