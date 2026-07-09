from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib

import numpy as np

from AEGIS.detector_artifact import DetectorArtifact
from AEGIS.guardrail import EmbeddingProvider, GuardrailPolicy, GuardrailRequest, GuardrailRuntime


@dataclass(frozen=True)
class ContractCheck:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_provider(spec: str) -> EmbeddingProvider:
    """Load a provider from `module:object`.

    The object can be an instance, class with a no-argument constructor, or factory
    function returning an object that satisfies the `EmbeddingProvider` protocol.
    """

    if ":" not in spec:
        raise ValueError("Provider spec must use 'module:object'.")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, object_name)
    if isinstance(value, type):
        return value()
    if callable(value) and not all(hasattr(value, attr) for attr in _PROVIDER_ATTRS):
        return value()
    return value


def validate_provider_contract(
    provider: EmbeddingProvider,
    artifact: DetectorArtifact,
    requests: list[GuardrailRequest] | None = None,
    require_matching_provenance: bool = True,
    require_deterministic: bool = True,
    required_modalities: tuple[str, ...] = (),
) -> dict[str, object]:
    requests = requests or [
        GuardrailRequest(text="Summarize the image.", request_id="contract-text"),
    ]
    checks: list[ContractCheck] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append(ContractCheck(name=name, ok=bool(ok), detail=detail))

    for attr in _PROVIDER_ATTRS:
        check(
            f"provider_has_{attr}",
            hasattr(provider, attr),
            f"{attr}={getattr(provider, attr, None)!r}",
        )

    provider_dim = int(getattr(provider, "feature_dim", -1))
    check(
        "feature_dim_matches_detector",
        provider_dim == artifact.feature_dim,
        f"provider={provider_dim}, detector={artifact.feature_dim}",
    )

    if require_matching_provenance:
        for attr in ("model_family", "model_id", "pooling"):
            check(
                f"{attr}_matches_detector",
                getattr(provider, attr, None) == getattr(artifact, attr),
                f"provider={getattr(provider, attr, None)!r}, detector={getattr(artifact, attr)!r}",
            )

    vectors = []
    embedded_requests: list[tuple[GuardrailRequest, np.ndarray, str]] = []
    request_records: list[dict[str, object]] = []
    for request in requests:
        request_label = request.request_id or str(len(request_records) + 1)
        request_records.append(
            {
                "request_id": request.request_id,
                "modality": request.modality,
                "has_text": bool(request.text.strip()),
                "n_images": len(request.image_paths),
            }
        )
        try:
            vector = np.asarray(provider.embed(request), dtype=np.float64)
            if vector.ndim == 2 and vector.shape[0] == 1:
                vector = vector.reshape(-1)
            vectors.append(vector)
            embedded_requests.append((request, vector, request_label))
            check(
                f"embed_{request_label}_shape",
                vector.shape == (artifact.feature_dim,),
                f"shape={vector.shape}, expected=({artifact.feature_dim},)",
            )
            check(
                f"embed_{request_label}_finite",
                bool(np.all(np.isfinite(vector))),
                "feature vector values are finite",
            )
        except Exception as exc:
            check(f"embed_{request_label}", False, str(exc))

    if require_deterministic:
        for request, first_vector, request_label in embedded_requests:
            try:
                first = first_vector.reshape(-1)
                second = np.asarray(provider.embed(request), dtype=np.float64).reshape(-1)
                check(
                    f"deterministic_embedding_{request_label}",
                    bool(np.allclose(first, second)),
                    "same request produced matching features",
                )
            except Exception as exc:
                check(f"deterministic_embedding_{request_label}", False, str(exc))

    modalities = sorted({str(record["modality"]) for record in request_records})
    for modality in required_modalities:
        check(
            f"required_modality_{modality}",
            modality in modalities,
            f"observed_modalities={modalities}",
        )

    try:
        runtime = GuardrailRuntime(
            artifact,
            policy=GuardrailPolicy(require_matching_provenance=require_matching_provenance),
        )
        if not vectors:
            raise ValueError("No provider feature vector available for runtime decision check.")
        decision = runtime.evaluate_request(requests[0], _CachedProvider(provider, vectors[0]))
        check(
            "runtime_decision",
            decision.action in {"allow", "review", "block"},
            f"action={decision.action}, verdict={decision.verdict}",
        )
    except Exception as exc:
        check("runtime_decision", False, str(exc))

    return {
        "ok": all(item.ok for item in checks),
        "provider": {
            "model_family": getattr(provider, "model_family", None),
            "model_id": getattr(provider, "model_id", None),
            "pooling": getattr(provider, "pooling", None),
            "feature_dim": getattr(provider, "feature_dim", None),
        },
        "detector": {
            "model_family": artifact.model_family,
            "model_id": artifact.model_id,
            "pooling": artifact.pooling,
            "feature_dim": artifact.feature_dim,
        },
        "request_coverage": {
            "n_requests": len(request_records),
            "modalities": modalities,
            "required_modalities": list(required_modalities),
            "requests": request_records,
        },
        "checks": [item.to_dict() for item in checks],
    }


_PROVIDER_ATTRS = ("model_family", "model_id", "pooling", "feature_dim")


class _CachedProvider:
    def __init__(self, provider: EmbeddingProvider, vector: np.ndarray) -> None:
        for attr in _PROVIDER_ATTRS:
            setattr(self, attr, getattr(provider, attr))
        self._vector = vector

    def embed(self, request: GuardrailRequest) -> np.ndarray:
        return self._vector
