from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from AEGIS.detector_artifact import (
    DetectorArtifact,
    load_detector_artifact_snapshot,
)
from AEGIS.provider_contract import FUSED_COMPONENT_POOLINGS, FUSED_TEXT_IMAGE_POOLING


DUAL_OR_MODE = "or"
FUSED_POOLING = FUSED_TEXT_IMAGE_POOLING
HEAD_POOLINGS = dict(zip(("text", "image"), FUSED_COMPONENT_POOLINGS, strict=True))


@dataclass(frozen=True)
class DetectorHeadConfig:
    name: str
    artifact_path: Path
    review_threshold: float


@dataclass(frozen=True)
class DetectorHead:
    name: str
    artifact: DetectorArtifact
    review_threshold: float
    artifact_sha256: str

    def __post_init__(self) -> None:
        expected_pooling = HEAD_POOLINGS.get(self.name)
        if expected_pooling is None:
            raise ValueError(f"Unsupported detector head name: {self.name!r}.")
        if self.artifact.pooling != expected_pooling:
            raise ValueError(
                f"Detector head {self.name!r} requires pooling {expected_pooling!r}; "
                f"received {self.artifact.pooling!r}."
            )
        review = float(self.review_threshold)
        if not np.isfinite(review) or not 0.0 <= review < self.artifact.threshold:
            raise ValueError(
                f"Detector head {self.name!r} review_threshold must be finite, "
                "non-negative, and below its artifact block threshold."
            )
        if (
            self.artifact_sha256 != self.artifact_sha256.lower()
            or len(self.artifact_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.artifact_sha256)
        ):
            raise ValueError(
                "Detector-head SHA-256 must be canonical lowercase hexadecimal."
            )


@dataclass(frozen=True)
class OrDetector:
    heads: tuple[DetectorHead, ...]
    mode: str = DUAL_OR_MODE

    def __post_init__(self) -> None:
        if self.mode != DUAL_OR_MODE:
            raise ValueError("Only the 'or' detector-set mode is supported.")
        by_name = {head.name: head for head in self.heads}
        if len(by_name) != len(self.heads):
            raise ValueError("Detector-set head names must be unique.")
        if set(by_name) != set(HEAD_POOLINGS):
            raise ValueError("An OR detector set requires exactly 'text' and 'image' heads.")
        common_names = (
            "model_family",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "preprocessing_sha256",
            "layer",
        )
        ordered = self.ordered_heads
        reference = ordered[0].artifact
        for head in ordered[1:]:
            mismatches = [
                name
                for name in common_names
                if getattr(head.artifact, name) != getattr(reference, name)
            ]
            if mismatches:
                raise ValueError(
                    "Detector heads have incompatible embedding provenance: "
                    + ", ".join(mismatches)
                    + "."
                )
            if head.artifact.feature_dim != reference.feature_dim:
                raise ValueError("Detector heads must have the same base feature dimension.")

    @property
    def ordered_heads(self) -> tuple[DetectorHead, ...]:
        by_name = {head.name: head for head in self.heads}
        return (by_name["text"], by_name["image"])

    @property
    def model_family(self) -> str:
        return self.ordered_heads[0].artifact.model_family

    @property
    def model_id(self) -> str:
        return self.ordered_heads[0].artifact.model_id

    @property
    def model_revision(self) -> str:
        return self.ordered_heads[0].artifact.model_revision

    @property
    def tokenizer_revision(self) -> str:
        return self.ordered_heads[0].artifact.tokenizer_revision

    @property
    def preprocessing_sha256(self) -> str:
        return self.ordered_heads[0].artifact.preprocessing_sha256

    @property
    def layer(self) -> int:
        return self.ordered_heads[0].artifact.layer

    @property
    def pooling(self) -> str:
        return FUSED_POOLING

    @property
    def base_feature_dim(self) -> int:
        return self.ordered_heads[0].artifact.feature_dim

    @property
    def feature_dim(self) -> int:
        return sum(head.artifact.feature_dim for head in self.ordered_heads)

    @property
    def source(self) -> str:
        return "or(" + ",".join(head.artifact.source for head in self.ordered_heads) + ")"

    @property
    def identity_sha256(self) -> str:
        return detector_set_identity_sha256(
            tuple(
                (head.name, head.artifact_sha256, head.review_threshold)
                for head in self.heads
            )
        )

    def split_features(self, features: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected fused features with shape (n, {self.feature_dim}), got {x.shape}."
            )
        boundary = self.ordered_heads[0].artifact.feature_dim
        return {"text": x[:, :boundary], "image": x[:, boundary:]}

    def score_heads(self, features: np.ndarray) -> dict[str, np.ndarray]:
        components = self.split_features(features)
        return {
            head.name: head.artifact.score(components[head.name])
            for head in self.ordered_heads
        }

    def status(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "identity_sha256": self.identity_sha256,
            "heads": [
                {
                    "name": head.name,
                    "artifact_version": 1,
                    "artifact_sha256": head.artifact_sha256,
                    "pooling": head.artifact.pooling,
                    "feature_dim": head.artifact.feature_dim,
                    "block_threshold": float(head.artifact.threshold),
                    "review_threshold": float(head.review_threshold),
                    "source": head.artifact.source,
                }
                for head in self.ordered_heads
            ],
        }


def detector_set_identity_sha256(
    heads: tuple[tuple[str, str, float], ...],
) -> str:
    normalized: list[tuple[str, str, float]] = []
    seen_names: set[str] = set()
    for name, artifact_sha256, review_threshold in heads:
        if name not in HEAD_POOLINGS or name in seen_names:
            raise ValueError("Detector identity requires unique supported head names.")
        if (
            artifact_sha256 != artifact_sha256.lower()
            or len(artifact_sha256) != 64
            or any(value not in "0123456789abcdef" for value in artifact_sha256)
        ):
            raise ValueError(
                "Detector-head SHA-256 must be canonical lowercase hexadecimal."
            )
        if isinstance(review_threshold, bool):
            raise ValueError("Detector-head review threshold must be a finite number.")
        review = float(review_threshold)
        if not np.isfinite(review):
            raise ValueError("Detector-head review threshold must be a finite number.")
        seen_names.add(name)
        normalized.append((name, artifact_sha256, review))
    if seen_names != set(HEAD_POOLINGS):
        raise ValueError("Detector identity requires exactly 'text' and 'image' heads.")
    payload = {
        "schema_version": 1,
        "mode": DUAL_OR_MODE,
        "heads": [
            {
                "name": name,
                "artifact_sha256": artifact_sha256,
                "review_threshold_hex": float(review_threshold).hex(),
            }
            for name, artifact_sha256, review_threshold in sorted(
                normalized, key=lambda item: item[0]
            )
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_or_detector(configs: tuple[DetectorHeadConfig, ...]) -> OrDetector:
    heads: list[DetectorHead] = []
    for config in configs:
        artifact, artifact_sha256 = load_detector_artifact_snapshot(config.artifact_path)
        heads.append(
            DetectorHead(
                name=config.name,
                artifact=artifact,
                review_threshold=float(config.review_threshold),
                artifact_sha256=artifact_sha256,
            )
        )
    return OrDetector(tuple(heads))


def load_detector_definition(
    artifact_path: str | Path | None,
    detector_heads: tuple[DetectorHeadConfig, ...] = (),
) -> DetectorArtifact | OrDetector:
    if detector_heads:
        if artifact_path is not None:
            raise ValueError("Configure either one detector artifact or an OR detector set.")
        return load_or_detector(detector_heads)
    if artifact_path is None:
        raise ValueError("A detector artifact path is required.")
    artifact, artifact_sha256 = load_detector_artifact_snapshot(artifact_path)
    artifact.artifact_sha256 = artifact_sha256
    return artifact
