from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping


TRAINING_IDENTITY_SCHEMA_VERSION = 1
DETECTOR_CORPUS_VERSION = "bordair_ocr_multichannel_counterfactual_v7"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def training_identity_payload(
    training_configuration: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    if not training_configuration:
        raise ValueError("training_configuration must be a non-empty mapping")
    if not selected_candidate:
        raise ValueError("selected_candidate must be a non-empty mapping")
    return {
        "schema_version": TRAINING_IDENTITY_SCHEMA_VERSION,
        "training_configuration": dict(training_configuration),
        "selected_candidate": dict(selected_candidate),
    }


def training_identity_sha256(
    training_configuration: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
) -> str:
    """Hash configuration and selected-candidate data using canonical JSON."""

    payload = training_identity_payload(training_configuration, selected_candidate)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def detector_source_label(
    *,
    corpus_version: str,
    corpus_manifest_sha256: str,
    target: str,
    pooling: str,
    training_identity_sha256_value: str,
) -> str:
    for name, value in (
        ("corpus_version", corpus_version),
        ("target", target),
        ("pooling", pooling),
    ):
        if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(f"{name} is not a safe source-label token: {value!r}")
    for name, value in (
        ("corpus_manifest_sha256", corpus_manifest_sha256),
        ("training_identity_sha256", training_identity_sha256_value),
    ):
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return (
        f"{corpus_version}_{corpus_manifest_sha256[:12]}_{target}_{pooling}_"
        f"cfgcand-{training_identity_sha256_value}"
    )


def validate_summary_training_identity(
    summary: Mapping[str, Any],
    *,
    target: str,
    pooling: str,
    corpus_manifest_sha256: str,
    artifact_source: str,
) -> str:
    configuration = summary.get("training_configuration")
    selected_candidate = summary.get("selected_candidate")
    identity = summary.get("training_identity")
    if not isinstance(configuration, Mapping):
        raise ValueError("Training summary has no training_configuration mapping.")
    if not isinstance(selected_candidate, Mapping):
        raise ValueError("Training summary has no selected_candidate mapping.")
    if not isinstance(identity, Mapping):
        raise ValueError("Training summary has no training_identity mapping.")
    digest = training_identity_sha256(configuration, selected_candidate)
    expected_identity = {
        "schema_version": TRAINING_IDENTITY_SCHEMA_VERSION,
        "sha256": digest,
    }
    if dict(identity) != expected_identity:
        raise ValueError("Training-summary identity digest does not match its payload.")
    expected_source = detector_source_label(
        corpus_version=DETECTOR_CORPUS_VERSION,
        corpus_manifest_sha256=corpus_manifest_sha256,
        target=target,
        pooling=pooling,
        training_identity_sha256_value=digest,
    )
    if artifact_source != expected_source:
        raise ValueError(
            "Artifact source label is not bound to the training configuration and "
            "selected candidate."
        )
    return digest
