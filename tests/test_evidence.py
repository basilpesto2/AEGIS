from __future__ import annotations

import json

import pandas as pd

from AEGIS.evidence import build_evidence_report, render_markdown_report


def test_evidence_report_marks_universal_claim_partial(tmp_path) -> None:
    (tmp_path / "models/aegis").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "data/processed/aegis_classify_multimodal_litmus_v1").mkdir(parents=True)
    (tmp_path / "models/aegis/aegis_qwen25vl3b_text_detector.npz").write_bytes(b"fake")
    (tmp_path / "docs/reproducibility_manifest.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )
    (tmp_path / "docs/production_guardrail_playbook.md").write_text("playbook", encoding="utf-8")
    (tmp_path / "configs/guardrail_policy.example.json").write_text("{}", encoding="utf-8")
    (tmp_path / "outputs/aegis_classify_multimodal_litmus_v1_summary.json").write_text(
        json.dumps({"n_samples": 32, "detector": {"model_family": "qwen25_vl"}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "label": ["benign", "malicious"],
            "attack_style": ["none", "role_play"],
        }
    ).to_csv(
        tmp_path / "data/processed/aegis_classify_multimodal_litmus_v1/metadata.csv",
        index=False,
    )

    report = build_evidence_report(tmp_path)
    markdown = render_markdown_report(report)

    assert report["overall_status"] in {"partial", "incomplete"}
    universal = [
        item for item in report["requirements"] if item["key"] == "universal_any_llm_mllm_claim"
    ][0]
    assert universal["status"] == "partial"
    assert "AEGIS Production Evidence Report" in markdown
