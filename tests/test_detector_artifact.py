from __future__ import annotations

import numpy as np

from AEGIS.detector_artifact import DetectorArtifact, load_detector_artifact, save_detector_artifact
from AEGIS.logistic import LogisticRegressionNumpy
from AEGIS.readiness import doctor, inspect_detector_artifact


def _fit_classifier() -> LogisticRegressionNumpy:
    features = np.asarray(
        [
            [-2.0, -1.0],
            [-1.0, -1.0],
            [-1.5, -0.5],
            [1.0, 1.0],
            [1.5, 0.5],
            [2.0, 1.0],
        ]
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    return LogisticRegressionNumpy(epochs=120, learning_rate=0.2, l2=0.0).fit(
        features,
        labels,
    )


def test_detector_artifact_round_trips_and_inspects(tmp_path) -> None:
    artifact = DetectorArtifact(
        classifier=_fit_classifier(),
        threshold=0.5,
        model_family="qwen25_vl",
        model_id="unit/model",
        layer=-1,
        pooling="text_tokens",
        source="unit-test",
    )
    path = save_detector_artifact(tmp_path / "detector.npz", artifact)

    loaded = load_detector_artifact(path)
    features = np.asarray([[-1.0, -1.0], [2.0, 1.0]])

    assert loaded.model_id == "unit/model"
    assert loaded.feature_dim == 2
    assert loaded.classify(features)[0]["recommended_action"] == "allow"
    assert loaded.classify(features)[1]["recommended_action"] == "block"
    assert loaded.score(features).tolist() == artifact.score(features).tolist()

    info = inspect_detector_artifact(path)
    assert info["feature_dim"] == 2
    assert info["pooling"] == "text_tokens"
    assert len(str(info["sha256"])) == 64

    report = doctor(path, cache_dir=tmp_path)
    assert report["ok"] is True
    assert all(check["ok"] for check in report["checks"])
