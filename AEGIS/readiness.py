from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from AEGIS.detector_artifact import ARTIFACT_VERSION, load_detector_artifact


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_detector_artifact(path: str | Path) -> dict[str, object]:
    artifact_path = Path(path)
    artifact = load_detector_artifact(artifact_path)
    classifier = artifact.classifier
    assert classifier.weights_ is not None
    assert classifier.mean_ is not None
    assert classifier.scale_ is not None

    return {
        "path": str(artifact_path),
        "exists": artifact_path.exists(),
        "sha256": sha256_file(artifact_path),
        "bytes": int(artifact_path.stat().st_size),
        "artifact_version": ARTIFACT_VERSION,
        "model_family": artifact.model_family,
        "model_id": artifact.model_id,
        "layer": artifact.layer,
        "pooling": artifact.pooling,
        "feature_dim": artifact.feature_dim,
        "threshold": float(artifact.threshold),
        "uncertainty_margin": float(artifact.uncertainty_margin),
        "source": artifact.source,
        "learning_rate": float(classifier.learning_rate),
        "epochs": int(classifier.epochs),
        "l2": float(classifier.l2),
        "standardize": bool(classifier.standardize),
        "random_seed": int(classifier.random_seed),
    }


def doctor(
    detector_path: str | Path,
    cache_dir: str | Path | None = None,
    require_cache_dir: bool = False,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def add_check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    try:
        info = inspect_detector_artifact(detector_path)
        add_check("detector_loadable", True, "Detector artifact loaded without pickle.")
        add_check(
            "feature_dim_positive",
            int(info["feature_dim"]) > 0,
            f"feature_dim={info['feature_dim']}",
        )
        add_check(
            "threshold_range",
            0.0 < float(info["threshold"]) < 1.0,
            f"threshold={info['threshold']}",
        )
        add_check(
            "uncertainty_margin_range",
            0.0 <= float(info["uncertainty_margin"]) < 0.5,
            f"uncertainty_margin={info['uncertainty_margin']}",
        )

        artifact = load_detector_artifact(detector_path)
        classifier = artifact.classifier
        arrays = [classifier.weights_, classifier.mean_, classifier.scale_]
        finite = all(value is not None and np.all(np.isfinite(value)) for value in arrays)
        add_check("finite_parameters", finite, "weights, mean, and scale are finite.")
        positive_scale = classifier.scale_ is not None and bool(np.all(classifier.scale_ > 0.0))
        add_check("positive_scale", positive_scale, "standardization scale values are positive.")
    except Exception as exc:  # pragma: no cover - exercised through CLI behavior.
        info = None
        add_check("detector_loadable", False, str(exc))

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_ok = cache_path.exists() and cache_path.is_dir()
        severity_ok = cache_ok or not require_cache_dir
        detail = f"cache_dir={cache_path}"
        if not cache_ok and not require_cache_dir:
            detail += " (not required for artifact-only checks)"
        add_check("model_cache_dir", severity_ok, detail)

    return {
        "ok": all(bool(check["ok"]) for check in checks),
        "detector": info,
        "checks": checks,
    }
