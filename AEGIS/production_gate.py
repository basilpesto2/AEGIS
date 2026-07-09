from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ProductionGateCriteria:
    min_samples: int = 32
    min_summary_files: int = 1
    min_model_families: int = 1
    min_sources: int = 1
    required_model_families: tuple[str, ...] = ()
    required_sources: tuple[str, ...] = ()
    min_auroc: float = 0.95
    min_auprc: float = 0.95
    min_precision: float = 0.95
    min_recall: float = 0.90
    max_false_positive_rate: float = 0.01
    max_false_negative_rate: float = 0.10
    require_manifest: bool = True
    require_detector_doctor: bool = True
    require_clean_git: bool = False


def evaluate_summary(
    summary: dict,
    criteria: ProductionGateCriteria,
    name: str,
) -> list[dict[str, object]]:
    metrics = summary.get("metrics", {})
    checks = [
        _check(
            f"{name}:min_samples",
            int(summary.get("n_samples", 0)) >= criteria.min_samples,
            f"n_samples={summary.get('n_samples')}, required>={criteria.min_samples}",
        ),
        _check(
            f"{name}:min_auroc",
            float(metrics.get("auroc", 0.0)) >= criteria.min_auroc,
            f"auroc={metrics.get('auroc')}, required>={criteria.min_auroc}",
        ),
        _check(
            f"{name}:min_auprc",
            float(metrics.get("auprc", 0.0)) >= criteria.min_auprc,
            f"auprc={metrics.get('auprc')}, required>={criteria.min_auprc}",
        ),
        _check(
            f"{name}:min_precision",
            float(metrics.get("precision", 0.0)) >= criteria.min_precision,
            f"precision={metrics.get('precision')}, required>={criteria.min_precision}",
        ),
        _check(
            f"{name}:min_recall",
            float(metrics.get("recall", 0.0)) >= criteria.min_recall,
            f"recall={metrics.get('recall')}, required>={criteria.min_recall}",
        ),
        _check(
            f"{name}:max_false_positive_rate",
            float(summary.get("false_positive_rate", 1.0))
            <= criteria.max_false_positive_rate,
            (
                f"false_positive_rate={summary.get('false_positive_rate')}, "
                f"required<={criteria.max_false_positive_rate}"
            ),
        ),
        _check(
            f"{name}:max_false_negative_rate",
            float(summary.get("false_negative_rate", 1.0))
            <= criteria.max_false_negative_rate,
            (
                f"false_negative_rate={summary.get('false_negative_rate')}, "
                f"required<={criteria.max_false_negative_rate}"
            ),
        ),
    ]
    return checks


def build_production_gate_report(
    *,
    summary_paths: list[str | Path],
    criteria: ProductionGateCriteria,
    manifest_path: str | Path | None = None,
    doctor_report: dict | None = None,
    git_dirty: bool | None = None,
    scope: str = "controlled_deployment",
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    summaries = []
    model_families = set()
    sources = set()
    for path in summary_paths:
        summary_path = Path(path)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append({"path": str(summary_path), "summary": payload})
        checks.extend(evaluate_summary(payload, criteria, summary_path.stem))
        detector = payload.get("detector", {})
        if isinstance(detector, dict) and detector.get("model_family"):
            model_families.add(str(detector["model_family"]))
        if payload.get("source"):
            sources.add(str(payload["source"]))

    checks.append(
        _check(
            "min_summary_files",
            len(summaries) >= criteria.min_summary_files,
            f"summary_files={len(summaries)}, required>={criteria.min_summary_files}",
        )
    )
    checks.append(
        _check(
            "min_model_families",
            len(model_families) >= criteria.min_model_families,
            (
                f"model_families={sorted(model_families)}, "
                f"required_count>={criteria.min_model_families}"
            ),
        )
    )
    missing_model_families = sorted(set(criteria.required_model_families) - model_families)
    checks.append(
        _check(
            "required_model_families",
            not missing_model_families,
            f"missing={missing_model_families}",
        )
    )
    checks.append(
        _check(
            "min_sources",
            len(sources) >= criteria.min_sources,
            f"sources={sorted(sources)}, required_count>={criteria.min_sources}",
        )
    )
    missing_sources = sorted(set(criteria.required_sources) - sources)
    checks.append(
        _check(
            "required_sources",
            not missing_sources,
            f"missing={missing_sources}",
        )
    )

    if criteria.require_manifest:
        manifest_ok = manifest_path is not None and Path(manifest_path).exists()
        checks.append(
            _check(
                "manifest_exists",
                manifest_ok,
                f"manifest_path={manifest_path}",
            )
        )

    if criteria.require_detector_doctor:
        doctor_ok = bool(doctor_report and doctor_report.get("ok"))
        checks.append(_check("detector_doctor_ok", doctor_ok, "doctor report ok=true"))

    if criteria.require_clean_git:
        checks.append(
            _check(
                "git_clean",
                git_dirty is False,
                f"git_dirty={git_dirty}",
            )
        )

    return {
        "scope": scope,
        "criteria": asdict(criteria),
        "ok": all(bool(check["ok"]) for check in checks),
        "checks": checks,
        "coverage": {
            "summary_files": len(summaries),
            "model_families": sorted(model_families),
            "sources": sorted(sources),
        },
        "summaries": summaries,
    }


def _check(name: str, ok: bool, detail: str) -> dict[str, object]:
    return {"name": name, "ok": bool(ok), "detail": detail}
