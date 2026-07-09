from __future__ import annotations

import numpy as np

from AEGIS.detector_artifact import DetectorArtifact
from AEGIS.litmus import LITMUS_PANEL_VERSION, build_litmus_panel, evaluate_litmus_predictions
from AEGIS.logistic import LogisticRegressionNumpy


def test_expanded_litmus_panel_is_balanced_and_multimodal(tmp_path) -> None:
    table = build_litmus_panel(tmp_path)

    assert len(table) == 32
    assert table["label"].value_counts().to_dict() == {"benign": 16, "malicious": 16}
    assert set(table["modality"]) == {"image_text"}
    assert set(table["panel_version"]) == {LITMUS_PANEL_VERSION}
    assert table["sample_id"].is_unique
    assert table["normalized_prompt_sha256"].is_unique
    assert all((tmp_path / image_path).exists() for image_path in table["image_path"])


def test_litmus_evaluation_reports_errors_and_uncertainty(tmp_path) -> None:
    table = build_litmus_panel(tmp_path)
    features = (table["label"] == "malicious").astype(float).to_numpy().reshape(-1, 1)
    classifier = LogisticRegressionNumpy(standardize=True)
    classifier.weights_ = np.asarray([10.0])
    classifier.bias_ = -5.0
    classifier.mean_ = np.asarray([0.0])
    classifier.scale_ = np.asarray([1.0])
    artifact = DetectorArtifact(
        classifier=classifier,
        threshold=0.5,
        model_family="qwen25_vl",
        model_id="unit/model",
        layer=-1,
        pooling="text_tokens",
        source="unit-litmus",
    )

    evaluation = evaluate_litmus_predictions(artifact, features, table)

    assert evaluation.summary["panel_version"] == LITMUS_PANEL_VERSION
    assert evaluation.summary["n_samples"] == 32
    assert evaluation.summary["perfect"] is True
    assert evaluation.summary["misclassified_sample_ids"] == []
