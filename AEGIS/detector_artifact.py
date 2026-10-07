from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path

import numpy as np

from AEGIS.logistic import LogisticRegressionNumpy
from AEGIS.provenance import is_sha256


ARTIFACT_VERSION = 1
SUPPORTED_MODEL_FAMILIES = {
    "qwen25_vl",
    "llava_onevision",
    "generic",
    "custom",
}


@dataclass
class DetectorArtifact:
    classifier: LogisticRegressionNumpy
    threshold: float
    model_family: str
    model_id: str
    layer: int
    pooling: str
    uncertainty_margin: float = 0.05
    source: str = ""
    model_revision: str = ""
    tokenizer_revision: str = ""
    preprocessing_sha256: str = ""

    def __post_init__(self) -> None:
        if self.model_family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(f"Unsupported model family: {self.model_family!r}.")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if not 0.0 <= self.uncertainty_margin < 0.5:
            raise ValueError("uncertainty_margin must be in [0, 0.5).")
        if self.preprocessing_sha256 and not is_sha256(self.preprocessing_sha256):
            raise ValueError("preprocessing_sha256 must be an empty string or SHA-256 hex.")
        _require_fitted_classifier(self.classifier)

    @property
    def feature_dim(self) -> int:
        assert self.classifier.weights_ is not None
        return int(self.classifier.weights_.shape[0])

    def score(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features with shape (n, {self.feature_dim}), got {x.shape}."
            )
        return self.classifier.predict_proba(x)

    def classify(self, features: np.ndarray) -> list[dict[str, object]]:
        scores = self.score(features)
        results: list[dict[str, object]] = []
        for score in scores:
            value = float(score)
            malicious = value >= self.threshold
            uncertain = abs(value - self.threshold) <= self.uncertainty_margin
            results.append(
                {
                    "verdict": "malicious" if malicious else "benign",
                    "risk_score": value,
                    "threshold": float(self.threshold),
                    "uncertain": bool(uncertain),
                    "recommended_action": (
                        "review" if uncertain else "block" if malicious else "allow"
                    ),
                }
            )
        return results


def save_detector_artifact(path: str | Path, artifact: DetectorArtifact) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    classifier = artifact.classifier
    assert classifier.weights_ is not None
    assert classifier.mean_ is not None
    assert classifier.scale_ is not None
    np.savez(
        output,
        artifact_version=np.asarray([ARTIFACT_VERSION], dtype=np.int64),
        weights=classifier.weights_.astype(np.float64),
        bias=np.asarray([classifier.bias_], dtype=np.float64),
        mean=classifier.mean_.astype(np.float64),
        scale=classifier.scale_.astype(np.float64),
        learning_rate=np.asarray([classifier.learning_rate], dtype=np.float64),
        epochs=np.asarray([classifier.epochs], dtype=np.int64),
        l2=np.asarray([classifier.l2], dtype=np.float64),
        standardize=np.asarray([int(classifier.standardize)], dtype=np.int64),
        random_seed=np.asarray([classifier.random_seed], dtype=np.int64),
        threshold=np.asarray([artifact.threshold], dtype=np.float64),
        uncertainty_margin=np.asarray([artifact.uncertainty_margin], dtype=np.float64),
        model_family=np.asarray([artifact.model_family]),
        model_id=np.asarray([artifact.model_id]),
        layer=np.asarray([artifact.layer], dtype=np.int64),
        pooling=np.asarray([artifact.pooling]),
        source=np.asarray([artifact.source]),
        model_revision=np.asarray([artifact.model_revision]),
        tokenizer_revision=np.asarray([artifact.tokenizer_revision]),
        preprocessing_sha256=np.asarray([artifact.preprocessing_sha256]),
    )
    return output


def load_detector_artifact(path: str | Path) -> DetectorArtifact:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Detector artifact does not exist: {artifact_path}")
    artifact, _ = load_detector_artifact_snapshot(artifact_path)
    return artifact


def load_detector_artifact_snapshot(
    path: str | Path,
) -> tuple[DetectorArtifact, str]:
    """Load and identify one immutable byte snapshot of an artifact file."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Detector artifact does not exist: {artifact_path}")
    payload = artifact_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        version = _scalar_int(data, "artifact_version")
        if version != ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported detector artifact version {version}; expected {ARTIFACT_VERSION}."
            )
        classifier = LogisticRegressionNumpy(
            learning_rate=_scalar_float(data, "learning_rate"),
            epochs=_scalar_int(data, "epochs"),
            l2=_scalar_float(data, "l2"),
            standardize=bool(_scalar_int(data, "standardize")),
            random_seed=_scalar_int(data, "random_seed"),
        )
        classifier.weights_ = np.asarray(data["weights"], dtype=np.float64)
        classifier.bias_ = _scalar_float(data, "bias")
        classifier.mean_ = np.asarray(data["mean"], dtype=np.float64)
        classifier.scale_ = np.asarray(data["scale"], dtype=np.float64)
        artifact = DetectorArtifact(
            classifier=classifier,
            threshold=_scalar_float(data, "threshold"),
            uncertainty_margin=_scalar_float(data, "uncertainty_margin"),
            model_family=_scalar_string(data, "model_family"),
            model_id=_scalar_string(data, "model_id"),
            layer=_scalar_int(data, "layer"),
            pooling=_scalar_string(data, "pooling"),
            source=_scalar_string(data, "source"),
            model_revision=_optional_scalar_string(data, "model_revision"),
            tokenizer_revision=_optional_scalar_string(data, "tokenizer_revision"),
            preprocessing_sha256=_optional_scalar_string(data, "preprocessing_sha256"),
        )
    return artifact, digest


def infer_embedding_provenance(path: str | Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        model_id = _scalar_string(data, "model_id")
        if "model_family" in data:
            model_family = _scalar_string(data, "model_family")
        elif "llava" in model_id.lower():
            model_family = "llava_onevision"
        elif "qwen" in model_id.lower():
            model_family = "qwen25_vl"
        else:
            model_family = "generic"
        provenance = {
            "model_family": model_family,
            "model_id": model_id,
            "layer": _scalar_int(data, "layer"),
            "pooling": _scalar_string(data, "pooling"),
        }
        for key in ("model_revision", "tokenizer_revision", "preprocessing_sha256"):
            if key in data:
                provenance[key] = _scalar_string(data, key)
        return provenance


def _require_fitted_classifier(classifier: LogisticRegressionNumpy) -> None:
    arrays = (classifier.weights_, classifier.mean_, classifier.scale_)
    if any(value is None for value in arrays):
        raise ValueError("Detector classifier must be fitted before it can be saved.")
    assert classifier.weights_ is not None
    assert classifier.mean_ is not None
    assert classifier.scale_ is not None
    if classifier.weights_.ndim != 1:
        raise ValueError("Detector weights must be one-dimensional.")
    if classifier.mean_.shape != classifier.weights_.shape:
        raise ValueError("Detector mean shape does not match its weights.")
    if classifier.scale_.shape != classifier.weights_.shape:
        raise ValueError("Detector scale shape does not match its weights.")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("Detector parameters contain non-finite values.")


def _scalar_string(data, key: str) -> str:
    _require_key(data, key)
    values = np.asarray(data[key]).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"Detector field {key!r} must contain one value.")
    return str(values[0])


def _optional_scalar_string(data, key: str, default: str = "") -> str:
    return _scalar_string(data, key) if key in data else default


def _scalar_float(data, key: str) -> float:
    value = float(_scalar_string(data, key))
    if not np.isfinite(value):
        raise ValueError(f"Detector field {key!r} must be finite.")
    return value


def _scalar_int(data, key: str) -> int:
    return int(_scalar_string(data, key))


def _require_key(data, key: str) -> None:
    if key not in data:
        raise ValueError(f"Detector artifact is missing field {key!r}.")
