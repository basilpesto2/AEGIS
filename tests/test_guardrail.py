from __future__ import annotations

import numpy as np

from AEGIS.detector_artifact import DetectorArtifact, infer_embedding_provenance
from AEGIS.guardrail import (
    GuardrailPolicy,
    GuardrailRequest,
    GuardrailRuntime,
    decision_summary,
    decisions_to_frame,
)
from AEGIS.logistic import LogisticRegressionNumpy


class StaticProvider:
    model_family = "generic"
    model_id = "unit-embedder"
    pooling = "text_tokens"
    feature_dim = 2

    def __init__(self, features: np.ndarray) -> None:
        self.features = features

    def embed(self, request: GuardrailRequest) -> np.ndarray:
        return self.features


def _artifact() -> DetectorArtifact:
    classifier = LogisticRegressionNumpy(standardize=False)
    classifier.weights_ = np.asarray([8.0, 0.0])
    classifier.bias_ = -4.0
    classifier.mean_ = np.asarray([0.0, 0.0])
    classifier.scale_ = np.asarray([1.0, 1.0])
    return DetectorArtifact(
        classifier=classifier,
        threshold=0.5,
        model_family="generic",
        model_id="unit-embedder",
        layer=-1,
        pooling="text_tokens",
        uncertainty_margin=0.05,
        source="unit",
    )


def test_guardrail_runtime_scores_features_without_raw_prompt_logging() -> None:
    runtime = GuardrailRuntime(_artifact())
    decisions = runtime.evaluate_features(
        np.asarray([[0.0, 1.0], [1.0, 1.0]]),
        sample_ids=["safe", "risk"],
        prompt_hashes=["hash-safe", "hash-risk"],
        modalities=["text", "text"],
    )

    assert decisions[0].action == "allow"
    assert decisions[1].action == "block"
    assert decisions[1].prompt_sha256 == "hash-risk"

    table = decisions_to_frame(decisions)
    summary = decision_summary(decisions)
    assert "text" not in table.columns
    assert summary["action_counts"] == {"allow": 1, "block": 1}


def test_guardrail_runtime_uses_provider_contract() -> None:
    runtime = GuardrailRuntime(
        _artifact(),
        policy=GuardrailPolicy(require_matching_provenance=True),
    )
    request = GuardrailRequest(text="hello", request_id="req-1")
    decision = runtime.evaluate_request(request, StaticProvider(np.asarray([1.0, 0.0])))

    assert decision.request_id == "req-1"
    assert decision.modality == "text"
    assert decision.action == "block"
    assert decision.prompt_sha256 is not None


def test_guardrail_runtime_fails_to_review_on_provider_mismatch() -> None:
    runtime = GuardrailRuntime(
        _artifact(),
        policy=GuardrailPolicy(require_matching_provenance=True, action_on_error="review"),
    )
    provider = StaticProvider(np.asarray([1.0, 0.0]))
    provider.model_id = "wrong-model"

    decision = runtime.evaluate_request(GuardrailRequest(text="hello"), provider)

    assert decision.action == "review"
    assert decision.verdict == "guardrail_error"
    assert decision.uncertain is True


def test_generic_embedding_provenance_is_supported(tmp_path) -> None:
    path = tmp_path / "features.npz"
    np.savez(
        path,
        embeddings=np.asarray([[0.0, 1.0]]),
        sample_id=np.asarray(["s0"]),
        model_family=np.asarray(["generic"]),
        model_id=np.asarray(["unit-embedder"]),
        layer=np.asarray([-1]),
        pooling=np.asarray(["text_tokens"]),
    )

    assert infer_embedding_provenance(path) == {
        "model_family": "generic",
        "model_id": "unit-embedder",
        "layer": -1,
        "pooling": "text_tokens",
    }
