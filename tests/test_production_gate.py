from __future__ import annotations

import json

from AEGIS.production_gate import (
    ProductionGateCriteria,
    build_production_gate_report,
    evaluate_summary,
)


def _summary() -> dict:
    return {
        "n_samples": 32,
        "source": "unit_source",
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0625,
        "detector": {"model_family": "qwen25_vl"},
        "metrics": {
            "auroc": 0.988,
            "auprc": 0.990,
            "precision": 1.0,
            "recall": 0.9375,
        },
    }


def test_production_gate_accepts_summary_within_thresholds(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
    manifest_path.write_text("{}", encoding="utf-8")

    report = build_production_gate_report(
        summary_paths=[summary_path],
        criteria=ProductionGateCriteria(),
        manifest_path=manifest_path,
        doctor_report={"ok": True},
        git_dirty=False,
    )

    assert report["ok"] is True
    assert report["coverage"]["model_families"] == ["qwen25_vl"]
    assert report["coverage"]["sources"] == ["unit_source"]


def test_production_gate_fails_low_recall() -> None:
    summary = _summary()
    summary["metrics"]["recall"] = 0.5
    checks = evaluate_summary(summary, ProductionGateCriteria(), "unit")

    failed = [check for check in checks if not check["ok"]]
    assert [check["name"] for check in failed] == ["unit:min_recall"]


def test_production_gate_can_require_model_family_coverage(tmp_path) -> None:
    summary = _summary()
    summary["detector"] = {"model_family": "qwen25_vl"}
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path.write_text("{}", encoding="utf-8")

    report = build_production_gate_report(
        summary_paths=[summary_path],
        criteria=ProductionGateCriteria(
            min_model_families=2,
            required_model_families=("qwen25_vl", "llava_onevision"),
            min_sources=2,
            required_sources=("unit_source", "heldout_source"),
        ),
        manifest_path=manifest_path,
        doctor_report={"ok": True},
        git_dirty=False,
    )

    failed = [check["name"] for check in report["checks"] if not check["ok"]]
    assert "min_model_families" in failed
    assert "required_model_families" in failed
    assert "min_sources" in failed
    assert "required_sources" in failed
