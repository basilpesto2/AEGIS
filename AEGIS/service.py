from __future__ import annotations

from typing import Any

import numpy as np

from AEGIS.detector_artifact import DetectorArtifact
from AEGIS.guardrail import (
    EmbeddingProvider,
    GuardrailPolicy,
    GuardrailRequest,
    GuardrailRuntime,
    decision_summary,
)


def evaluate_feature_payload(
    artifact: DetectorArtifact,
    payload: dict[str, Any],
    policy: GuardrailPolicy | None = None,
) -> dict[str, object]:
    """Evaluate a JSON-style payload containing feature vectors.

    Expected payload fields:
    - `features`: one feature vector or a list of feature vectors.
    - optional `sample_ids`, `request_ids`, `prompt_sha256`, `modalities`.
    """

    if "features" not in payload:
        raise ValueError("Payload must contain a 'features' field.")
    features = np.asarray(payload["features"], dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    runtime = GuardrailRuntime(artifact, policy=policy)
    decisions = runtime.evaluate_features(
        features,
        sample_ids=_optional_str_list(payload.get("sample_ids")),
        request_ids=_optional_str_list(payload.get("request_ids")),
        prompt_hashes=_optional_str_list(payload.get("prompt_sha256")),
        modalities=_optional_str_list(payload.get("modalities")),
    )
    return {
        "summary": decision_summary(decisions),
        "decisions": [decision.to_dict() for decision in decisions],
    }


def evaluate_request_payload(
    artifact: DetectorArtifact,
    provider: EmbeddingProvider,
    payload: dict[str, Any],
    policy: GuardrailPolicy | None = None,
) -> dict[str, object]:
    """Evaluate JSON-style prompt request payloads through an embedding provider.

    Expected payload fields for a single request:
    - optional `text`
    - optional `image_path` or `image_paths`
    - optional `request_id`
    - optional `metadata` object

    Batch payloads can use `requests: [{...}, {...}]`.
    """

    request_payloads = payload.get("requests")
    if request_payloads is None:
        requests = [_request_from_payload(payload)]
    else:
        if not isinstance(request_payloads, list) or not request_payloads:
            raise ValueError("'requests' must be a non-empty list when provided.")
        requests = [_request_from_payload(item) for item in request_payloads]

    runtime = GuardrailRuntime(artifact, policy=policy)
    decisions = [runtime.evaluate_request(request, provider) for request in requests]
    return {
        "summary": decision_summary(decisions),
        "decisions": [decision.to_dict() for decision in decisions],
    }


def _optional_str_list(value) -> list[str | None] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [None if item is None else str(item) for item in value]


def _request_from_payload(payload: dict[str, Any]) -> GuardrailRequest:
    if not isinstance(payload, dict):
        raise ValueError("Each request payload must be an object.")
    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("'metadata' must be an object when provided.")

    return GuardrailRequest(
        text="" if payload.get("text") is None else str(payload.get("text")),
        image_paths=_coerce_image_paths(payload),
        request_id=None if payload.get("request_id") is None else str(payload.get("request_id")),
        metadata=metadata,
    )


def _coerce_image_paths(payload: dict[str, Any]) -> tuple[str, ...]:
    value = payload.get("image_paths", payload.get("image_path", ()))
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list | tuple):
        raise ValueError("'image_paths' must be a string or list of strings.")
    return tuple(str(item) for item in value)
