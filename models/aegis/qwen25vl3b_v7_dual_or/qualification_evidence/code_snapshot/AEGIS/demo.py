from __future__ import annotations

import hashlib
import re
import time

import numpy as np

from AEGIS.detector_artifact import DetectorArtifact
from AEGIS.guardrail import GuardrailPolicy, GuardrailRequest, GuardrailRuntime
from AEGIS.logistic import LogisticRegressionNumpy


DEMO_WARNING = (
    "Functional onboarding demo only. This deterministic provider is not a trained "
    "safety model and must never be used for real policy decisions."
)


class DemoEmbeddingProvider:
    model_family = "generic"
    model_id = "aegis/functional-demo-v1"
    pooling = "demo_features"
    feature_dim = 8

    def embed(self, request: GuardrailRequest) -> np.ndarray:
        normalized = " ".join(request.text.lower().split())
        tokens = set(re.findall(r"[a-z0-9]+", normalized))
        risk_terms = {
            "bypass",
            "evade",
            "exfiltrate",
            "malware",
            "phishing",
            "steal",
        }
        safe_terms = {"summarize", "exercise", "weather", "recipe", "explain"}
        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
        return np.asarray(
            [
                float(bool(tokens & risk_terms)),
                float(len(tokens & risk_terms)),
                float(bool(tokens & safe_terms)),
                float(bool(request.image_paths)),
                len(normalized) / 1000.0,
                digest[0] / 2550.0,
                digest[1] / 2550.0,
                1.0,
            ],
            dtype=np.float32,
        )


class DelayedDemoEmbeddingProvider(DemoEmbeddingProvider):
    """Deterministic diagnostic provider for exercising hard worker timeouts."""

    def __init__(self, delay_seconds: float = 1.0) -> None:
        self.delay_seconds = float(delay_seconds)

    def embed(self, request: GuardrailRequest) -> np.ndarray:
        time.sleep(self.delay_seconds)
        return super().embed(request)


def demo_artifact() -> DetectorArtifact:
    classifier = LogisticRegressionNumpy(standardize=False)
    classifier.weights_ = np.asarray([8.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    classifier.bias_ = -4.0
    classifier.mean_ = np.zeros(8)
    classifier.scale_ = np.ones(8)
    return DetectorArtifact(
        classifier=classifier,
        threshold=0.5,
        model_family=DemoEmbeddingProvider.model_family,
        model_id=DemoEmbeddingProvider.model_id,
        layer=-1,
        pooling=DemoEmbeddingProvider.pooling,
        uncertainty_margin=0.05,
        source="functional_demo_not_for_safety",
    )


def evaluate_demo(text: str, *, request_id: str = "demo-request") -> dict[str, object]:
    if not text.strip():
        raise ValueError("Demo text cannot be empty.")
    runtime = GuardrailRuntime(
        demo_artifact(),
        policy=GuardrailPolicy(
            action_on_error="review",
            require_matching_provenance=True,
        ),
    )
    decision = runtime.evaluate_request(
        GuardrailRequest(text=text, request_id=request_id),
        DemoEmbeddingProvider(),
    )
    return {
        "warning": DEMO_WARNING,
        "decision": decision.to_dict(),
    }
