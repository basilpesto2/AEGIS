from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from AEGIS.detector_artifact import DetectorArtifact


GUARDRAIL_SCHEMA_VERSION = 1
VALID_ACTIONS = {"allow", "review", "block"}


class EmbeddingProvider(Protocol):
    """Protocol for plugging AEGIS into any LLM/MLLM embedding source."""

    model_family: str
    model_id: str
    pooling: str
    feature_dim: int

    def embed(self, request: "GuardrailRequest") -> np.ndarray:
        """Return one feature vector with shape `(feature_dim,)` or `(1, feature_dim)`."""


@dataclass(frozen=True)
class GuardrailRequest:
    text: str = ""
    image_paths: tuple[str, ...] = ()
    request_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def modality(self) -> str:
        has_text = bool(self.text.strip())
        has_image = bool(self.image_paths)
        if has_text and has_image:
            return "image_text"
        if has_image:
            return "image"
        return "text"

    def fingerprints(self, hash_images: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "request_id": self.request_id,
            "modality": self.modality,
            "prompt_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "n_images": len(self.image_paths),
        }
        if hash_images:
            payload["image_sha256"] = [
                _sha256_file(Path(path)) for path in self.image_paths
            ]
        return payload


@dataclass(frozen=True)
class GuardrailPolicy:
    """Threshold and failure behavior for production guardrail decisions."""

    block_threshold: float | None = None
    review_margin: float | None = None
    action_on_error: str = "review"
    require_matching_provenance: bool = False
    hash_images: bool = False

    def __post_init__(self) -> None:
        if self.block_threshold is not None and not 0.0 < self.block_threshold < 1.0:
            raise ValueError("block_threshold must be between 0 and 1.")
        if self.review_margin is not None and not 0.0 <= self.review_margin < 0.5:
            raise ValueError("review_margin must be in [0, 0.5).")
        if self.action_on_error not in VALID_ACTIONS:
            raise ValueError(f"action_on_error must be one of {sorted(VALID_ACTIONS)}.")


@dataclass(frozen=True)
class GuardrailDecision:
    action: str
    verdict: str
    risk_score: float
    threshold: float
    uncertain: bool
    reasons: tuple[str, ...]
    schema_version: int = GUARDRAIL_SCHEMA_VERSION
    request_id: str | None = None
    sample_id: str | None = None
    prompt_sha256: str | None = None
    image_sha256: tuple[str, ...] = ()
    modality: str | None = None
    model_family: str | None = None
    model_id: str | None = None
    pooling: str | None = None
    detector_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["image_sha256"] = list(self.image_sha256)
        return payload


class GuardrailRuntime:
    """Model-agnostic AEGIS decision engine for precomputed or provider features."""

    def __init__(
        self,
        artifact: DetectorArtifact,
        policy: GuardrailPolicy | None = None,
    ) -> None:
        self.artifact = artifact
        self.policy = policy or GuardrailPolicy()

    @property
    def threshold(self) -> float:
        return (
            float(self.policy.block_threshold)
            if self.policy.block_threshold is not None
            else float(self.artifact.threshold)
        )

    @property
    def review_margin(self) -> float:
        return (
            float(self.policy.review_margin)
            if self.policy.review_margin is not None
            else float(self.artifact.uncertainty_margin)
        )

    def evaluate_request(
        self,
        request: GuardrailRequest,
        provider: EmbeddingProvider,
    ) -> GuardrailDecision:
        try:
            self._check_provider(provider)
            features = np.asarray(provider.embed(request), dtype=np.float64)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            if features.shape[0] != 1:
                raise ValueError(f"Provider returned {features.shape[0]} feature rows for one request.")
            fingerprints = request.fingerprints(hash_images=self.policy.hash_images)
            return self.evaluate_features(
                features,
                request_ids=[request.request_id],
                prompt_hashes=[str(fingerprints["prompt_sha256"])],
                image_hashes=[
                    tuple(str(value) for value in fingerprints.get("image_sha256", []))
                ],
                modalities=[str(fingerprints["modality"])],
            )[0]
        except Exception:
            if self.policy.action_on_error == "allow":
                verdict = "unknown"
            else:
                verdict = "guardrail_error"
            return GuardrailDecision(
                action=self.policy.action_on_error,
                verdict=verdict,
                risk_score=float("nan"),
                threshold=self.threshold,
                uncertain=True,
                reasons=("embedding_or_scoring_error",),
                request_id=request.request_id,
                prompt_sha256=request.fingerprints()["prompt_sha256"],
                modality=request.modality,
                model_family=getattr(provider, "model_family", None),
                model_id=getattr(provider, "model_id", None),
                pooling=getattr(provider, "pooling", None),
                detector_source=self.artifact.source,
            )

    def evaluate_features(
        self,
        features: np.ndarray,
        sample_ids: list[str | None] | None = None,
        request_ids: list[str | None] | None = None,
        prompt_hashes: list[str | None] | None = None,
        image_hashes: list[tuple[str, ...]] | None = None,
        modalities: list[str | None] | None = None,
    ) -> list[GuardrailDecision]:
        x = np.asarray(features, dtype=np.float64)
        scores = self.artifact.score(x)
        n_rows = len(scores)
        sample_ids = _default_list(sample_ids, n_rows)
        request_ids = _default_list(request_ids, n_rows)
        prompt_hashes = _default_list(prompt_hashes, n_rows)
        image_hashes = image_hashes or [tuple() for _ in range(n_rows)]
        modalities = _default_list(modalities, n_rows)
        _require_length(sample_ids, n_rows, "sample_ids")
        _require_length(request_ids, n_rows, "request_ids")
        _require_length(prompt_hashes, n_rows, "prompt_hashes")
        _require_length(image_hashes, n_rows, "image_hashes")
        _require_length(modalities, n_rows, "modalities")

        decisions = []
        for idx, score in enumerate(scores):
            decisions.append(
                self._decision_from_score(
                    score=float(score),
                    sample_id=sample_ids[idx],
                    request_id=request_ids[idx],
                    prompt_sha256=prompt_hashes[idx],
                    image_sha256=image_hashes[idx],
                    modality=modalities[idx],
                )
            )
        return decisions

    def evaluate_metadata_features(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame | None = None,
        sample_ids: np.ndarray | None = None,
    ) -> list[GuardrailDecision]:
        n_rows = int(np.asarray(features).shape[0])
        ids = (
            [str(value) for value in sample_ids]
            if sample_ids is not None
            else _metadata_values(metadata, "sample_id", n_rows)
        )
        prompt_hashes = _metadata_hashes(metadata, "text", n_rows)
        modalities = _metadata_values(metadata, "modality", n_rows)
        return self.evaluate_features(
            features,
            sample_ids=ids,
            prompt_hashes=prompt_hashes,
            modalities=modalities,
        )

    def _decision_from_score(
        self,
        *,
        score: float,
        sample_id: str | None,
        request_id: str | None,
        prompt_sha256: str | None,
        image_sha256: tuple[str, ...],
        modality: str | None,
    ) -> GuardrailDecision:
        threshold = self.threshold
        malicious = score >= threshold
        uncertain = abs(score - threshold) <= self.review_margin
        if uncertain:
            action = "review"
        elif malicious:
            action = "block"
        else:
            action = "allow"

        reasons = []
        reasons.append("score_above_threshold" if malicious else "score_below_threshold")
        if uncertain:
            reasons.append("within_uncertainty_margin")

        return GuardrailDecision(
            action=action,
            verdict="malicious" if malicious else "benign",
            risk_score=score,
            threshold=threshold,
            uncertain=uncertain,
            reasons=tuple(reasons),
            request_id=request_id,
            sample_id=sample_id,
            prompt_sha256=prompt_sha256,
            image_sha256=image_sha256,
            modality=modality,
            model_family=self.artifact.model_family,
            model_id=self.artifact.model_id,
            pooling=self.artifact.pooling,
            detector_source=self.artifact.source,
        )

    def _check_provider(self, provider: EmbeddingProvider) -> None:
        if int(provider.feature_dim) != self.artifact.feature_dim:
            raise ValueError(
                f"Provider feature_dim {provider.feature_dim} does not match "
                f"detector feature_dim {self.artifact.feature_dim}."
            )
        if not self.policy.require_matching_provenance:
            return
        mismatches = []
        for name in ("model_family", "model_id", "pooling"):
            provider_value = getattr(provider, name)
            detector_value = getattr(self.artifact, name)
            if provider_value != detector_value:
                mismatches.append(f"{name}: {provider_value!r} != {detector_value!r}")
        if mismatches:
            raise ValueError("Provider provenance mismatch: " + "; ".join(mismatches))


def decisions_to_frame(decisions: list[GuardrailDecision]) -> pd.DataFrame:
    return pd.DataFrame([decision.to_dict() for decision in decisions])


def decision_summary(decisions: list[GuardrailDecision]) -> dict[str, object]:
    actions = {}
    verdicts = {}
    uncertain = 0
    for decision in decisions:
        actions[decision.action] = actions.get(decision.action, 0) + 1
        verdicts[decision.verdict] = verdicts.get(decision.verdict, 0) + 1
        uncertain += int(decision.uncertain)
    return {
        "schema_version": GUARDRAIL_SCHEMA_VERSION,
        "n_samples": len(decisions),
        "action_counts": actions,
        "verdict_counts": verdicts,
        "n_uncertain": uncertain,
    }


def _metadata_hashes(
    metadata: pd.DataFrame | None,
    column: str,
    n_rows: int,
) -> list[str | None]:
    if metadata is None or column not in metadata:
        return [None] * n_rows
    return [
        hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        for value in metadata[column].fillna("").astype(str)
    ]


def _metadata_values(
    metadata: pd.DataFrame | None,
    column: str,
    n_rows: int,
) -> list[str | None]:
    if metadata is None or column not in metadata:
        return [None] * n_rows
    return [str(value) for value in metadata[column].fillna("").astype(str)]


def _default_list(values: list | None, n_rows: int) -> list:
    return values if values is not None else [None] * n_rows


def _require_length(values: list, n_rows: int, name: str) -> None:
    if len(values) != n_rows:
        raise ValueError(f"{name} length {len(values)} does not match n_rows {n_rows}.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
