from __future__ import annotations

import json

import numpy as np
import pandas as pd

from AEGIS.calibration import CalibrationCriteria, calibrate_threshold, policy_payload
from AEGIS.detector_artifact import DetectorArtifact
from AEGIS.guardrail import GuardrailRequest
from AEGIS.logistic import LogisticRegressionNumpy
from AEGIS.monitoring import summarize_decisions
from AEGIS.service import evaluate_feature_payload, evaluate_request_payload


def _artifact() -> DetectorArtifact:
    classifier = LogisticRegressionNumpy(standardize=False)
    classifier.weights_ = np.asarray([6.0, 0.0])
    classifier.bias_ = -3.0
    classifier.mean_ = np.asarray([0.0, 0.0])
    classifier.scale_ = np.asarray([1.0, 1.0])
    return DetectorArtifact(
        classifier=classifier,
        threshold=0.5,
        model_family="generic",
        model_id="unit",
        layer=-1,
        pooling="text_tokens",
    )


class UnitProvider:
    model_family = "generic"
    model_id = "unit"
    pooling = "text_tokens"
    feature_dim = 2

    def embed(self, request: GuardrailRequest) -> np.ndarray:
        if "malicious" in request.text:
            return np.asarray([1.0, 0.0])
        return np.asarray([0.0, 0.0])


def test_calibration_selects_feasible_threshold_and_policy() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])

    calibration = calibrate_threshold(
        labels,
        scores,
        CalibrationCriteria(max_false_positive_rate=0.0, min_recall=1.0),
    )
    policy = policy_payload(calibration, detector_path="detector.npz", name="unit")

    assert calibration["feasible"] is True
    assert calibration["threshold"] == 0.8
    assert policy["policy"]["block_threshold"] == 0.8
    assert policy["logging"]["store_raw_prompts"] is False


def test_monitoring_summarizes_actions_and_drift(tmp_path) -> None:
    decisions = pd.DataFrame(
        {
            "action": ["allow", "review", "block"],
            "verdict": ["benign", "benign", "malicious"],
            "risk_score": [0.1, 0.55, 0.9],
            "uncertain": [False, True, False],
        }
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "n_samples": 3,
                "action_counts": {"allow": 2, "review": 1},
                "uncertain_rate": 1 / 3,
                "score_summary": {"mean": 0.3, "p95": 0.6},
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_decisions(decisions, baseline_path=baseline_path)

    assert summary["n_samples"] == 3
    assert summary["action_counts"] == {"allow": 1, "block": 1, "review": 1}
    assert "drift" in summary
    assert summary["drift"]["action_rate_delta"]["block"] == 1 / 3


def test_feature_payload_service_returns_decisions() -> None:
    response = evaluate_feature_payload(
        _artifact(),
        {
            "features": [[0.0, 0.0], [1.0, 0.0]],
            "sample_ids": ["a", "b"],
            "prompt_sha256": ["hash-a", "hash-b"],
            "modalities": ["text", "text"],
        },
    )

    assert response["summary"]["action_counts"] == {"allow": 1, "block": 1}
    assert response["decisions"][0]["sample_id"] == "a"
    assert response["decisions"][1]["prompt_sha256"] == "hash-b"


def test_request_payload_service_uses_provider_and_hashes_prompt() -> None:
    response = evaluate_request_payload(
        _artifact(),
        UnitProvider(),
        {"text": "benign request", "request_id": "r0"},
    )

    decision = response["decisions"][0]
    assert response["summary"]["action_counts"] == {"allow": 1}
    assert decision["request_id"] == "r0"
    assert decision["prompt_sha256"] is not None
    assert "benign request" not in json.dumps(response)


def test_request_payload_service_supports_batches() -> None:
    response = evaluate_request_payload(
        _artifact(),
        UnitProvider(),
        {
            "requests": [
                {"text": "ordinary request", "request_id": "r0"},
                {"text": "malicious request", "request_id": "r1"},
            ]
        },
    )

    assert response["summary"]["action_counts"] == {"allow": 1, "block": 1}
    assert [item["request_id"] for item in response["decisions"]] == ["r0", "r1"]
