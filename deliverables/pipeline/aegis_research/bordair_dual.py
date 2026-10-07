from __future__ import annotations

"""Durable contracts and validation-only selection for Bordair dual heads.

The pair keeps the existing version-1 ``DetectorArtifact`` format for each
channel.  The pair manifest is the only new format.  It binds the two artifact
bytes, their shared embedding provenance, the text-then-image cache layout, and
the OR decision policy without changing the historical fused workflow.
"""

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PAIR_MANIFEST_SCHEMA_VERSION = 1
PAIR_KIND = "aegis_detector_pair"
PAIR_COMPOSITION = "or"
CACHE_POOLING = "text_image_tokens"
CACHE_ORDERING = ("text_tokens", "image_tokens")
HEAD_POOLINGS = {"image": "image_tokens", "text": "text_tokens"}
ACTION_ORDER = {"allow": 0, "review": 1, "block": 2}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
FROZEN_PANEL_ORDER = ("regression", "text_led", "external_benign")
ONE_SHOT_SCHEMA_VERSION = 1
ONE_SHOT_KIND = "one_shot_frozen_dual_v1"
ONE_SHOT_CACHE_STATISTICS = {
    "hits": 50,
    "misses": 0,
    "invalid_entries": 0,
    "rebuilds": 0,
    "writes": 0,
    "write_errors": 0,
    "delegate_calls": 0,
}
ONE_SHOT_GATE_NAMES = {
    "external_benign_exact_counts",
    "external_benign_fewer_than_3_blocked",
    "fixed_benign_fewer_than_3_blocked",
    "fixed_exact_counts",
    "fixed_malicious_10_of_10_blocked",
    "text_led_exact_counts",
    "text_led_malicious_10_of_10_blocked",
}
ONE_SHOT_METADATA_FILES = {
    "regression": ("regression_metadata", "final_metadata_v7.csv"),
    "text_led": ("text_led_metadata", "text_led_final_metadata_v7.csv"),
    "external_benign": (
        "external_benign_metadata",
        "external_benign_metadata_v7.csv",
    ),
}


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualification_cache_locator(subject: str, panel: str, sample_id: str) -> str:
    """Return the path-independent logical locator for one frozen cache entry."""
    if not isinstance(subject, str) or SHA256_PATTERN.fullmatch(subject) is None:
        raise ValueError("qualification cache subject must be a lowercase SHA-256")
    if panel not in FROZEN_PANEL_ORDER:
        raise ValueError("qualification cache panel is invalid")
    if (not isinstance(sample_id, str) or not sample_id
            or any(character in sample_id for character in "/\\:\x00")
            or sample_id in {".", ".."}):
        raise ValueError("qualification cache sample ID is invalid")
    return f"{subject}/cache/{panel}/{sample_id}.npz"


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_probability(value: object, name: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok or result >= 1.0:
        relation = "[0, 1)" if allow_zero else "(0, 1)"
        raise ValueError(f"{name} must be finite and in {relation}")
    return result


def runtime_detector_identity_payload(
    heads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the minimal identity shared with the durable AEGIS runtime.

    The hexadecimal float spelling avoids platform-dependent decimal
    serialization.  Head order is canonical by name.
    """

    if set(heads) != set(HEAD_POOLINGS):
        raise ValueError("runtime identity requires exactly image and text heads")
    normalized: list[dict[str, str]] = []
    for name in sorted(heads):
        entry = heads[name]
        digest = _require_sha256(entry.get("sha256"), f"{name} artifact sha256")
        review = _finite_probability(
            entry.get("review_threshold"),
            f"{name} review threshold",
        )
        normalized.append(
            {
                "name": name,
                "artifact_sha256": digest,
                "review_threshold_hex": review.hex(),
            }
        )
    return {"schema_version": 1, "mode": "or", "heads": normalized}


def runtime_detector_identity_sha256(
    heads: Mapping[str, Mapping[str, Any]],
) -> str:
    return canonical_json_sha256(runtime_detector_identity_payload(heads))


def qualification_subject_payload(
    *,
    target: str,
    corpus_manifest_sha256: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    shared_provenance: Mapping[str, Any],
    runtime_detector_identity_sha256_value: str,
) -> dict[str, Any]:
    """Return the path-independent detector pair qualified by one-shot evidence."""

    if not isinstance(target, str) or not target:
        raise ValueError("qualification subject target is missing")
    if set(artifacts) != set(HEAD_POOLINGS):
        raise ValueError("qualification subject requires image and text artifacts")
    normalized_artifacts: dict[str, dict[str, Any]] = {}
    for name in sorted(HEAD_POOLINGS):
        entry = artifacts[name]
        block = _finite_probability(
            entry.get("block_threshold"),
            f"qualification {name} block threshold",
            allow_zero=False,
        )
        review = _finite_probability(
            entry.get("review_threshold"),
            f"qualification {name} review threshold",
        )
        if review >= block:
            raise ValueError(
                f"qualification {name} review threshold must be below block threshold"
            )
        feature_dim = entry.get("feature_dim")
        if (
            isinstance(feature_dim, bool)
            or not isinstance(feature_dim, int)
            or feature_dim <= 0
        ):
            raise ValueError(f"qualification {name} feature dimension is invalid")
        if entry.get("pooling") != HEAD_POOLINGS[name]:
            raise ValueError(f"qualification {name} pooling is invalid")
        normalized_artifacts[name] = {
            "sha256": _require_sha256(
                entry.get("sha256"), f"qualification {name} artifact sha256"
            ),
            "pooling": entry.get("pooling"),
            "feature_dim": feature_dim,
            "block_threshold_hex": block.hex(),
            "review_threshold_hex": review.hex(),
        }
    required_shared = (
        "model_family",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "preprocessing_sha256",
        "layer",
    )
    normalized_shared: dict[str, Any] = {}
    for name in required_shared:
        value = shared_provenance.get(name)
        if name == "layer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("qualification shared layer must be an integer")
        elif not isinstance(value, str) or not value:
            raise ValueError(f"qualification shared provenance {name} is missing")
        normalized_shared[name] = value
    _require_sha256(
        normalized_shared["preprocessing_sha256"],
        "qualification preprocessing sha256",
    )
    runtime_digest = _require_sha256(
        runtime_detector_identity_sha256_value,
        "qualification runtime detector identity",
    )
    if runtime_detector_identity_sha256(artifacts) != runtime_digest:
        raise ValueError("qualification runtime identity differs from its artifacts")
    return {
        "schema_version": 1,
        "target": target,
        "corpus_manifest_sha256": _require_sha256(
            corpus_manifest_sha256, "qualification corpus manifest sha256"
        ),
        "artifacts": normalized_artifacts,
        "shared_provenance": normalized_shared,
        "runtime_detector_identity_sha256": runtime_digest,
    }


def qualification_subject_sha256(**kwargs: Any) -> str:
    return canonical_json_sha256(qualification_subject_payload(**kwargs))


def pair_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the relocation-independent semantic identity of a pair."""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(HEAD_POOLINGS):
        raise ValueError("pair manifest requires exactly image and text artifacts")
    normalized_artifacts: dict[str, dict[str, Any]] = {}
    for name in sorted(HEAD_POOLINGS):
        entry = artifacts[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"pair artifact entry {name!r} must be an object")
        normalized_artifacts[name] = {
            "sha256": _require_sha256(entry.get("sha256"), f"{name} artifact sha256"),
            "pooling": entry.get("pooling"),
            "feature_dim": entry.get("feature_dim"),
            "block_threshold": entry.get("block_threshold"),
            "review_threshold": entry.get("review_threshold"),
            "source": entry.get("source"),
        }
    return {
        "schema_version": PAIR_MANIFEST_SCHEMA_VERSION,
        "kind": PAIR_KIND,
        "composition": PAIR_COMPOSITION,
        "target": manifest.get("target"),
        "training_identity_sha256": manifest.get("training_identity_sha256"),
        "corpus": manifest.get("corpus"),
        "shared_provenance": manifest.get("shared_provenance"),
        "cache_representation": manifest.get("cache_representation"),
        "development_qualification_evidence": manifest.get(
            "development_qualification_evidence"
        ),
        "qualification_evidence": manifest.get("qualification_evidence"),
        "artifacts": normalized_artifacts,
        "runtime_detector_identity_sha256": manifest.get(
            "runtime_detector_identity_sha256"
        ),
    }


def pair_identity_sha256(manifest: Mapping[str, Any]) -> str:
    return canonical_json_sha256(pair_identity_payload(manifest))


@dataclass(frozen=True)
class PairArtifactPaths:
    image: Path
    text: Path


@dataclass(frozen=True)
class LoadedDetectorPair:
    manifest: dict[str, Any]
    manifest_path: Path
    image_artifact: Any
    text_artifact: Any
    paths: PairArtifactPaths

    @property
    def base_feature_dim(self) -> int:
        return int(self.image_artifact.feature_dim)


def artifact_provenance(artifact: Any) -> dict[str, Any]:
    return {
        "model_family": artifact.model_family,
        "model_id": artifact.model_id,
        "model_revision": artifact.model_revision,
        "tokenizer_revision": artifact.tokenizer_revision,
        "preprocessing_sha256": artifact.preprocessing_sha256,
        "layer": int(artifact.layer),
    }


def artifact_entry(
    *,
    name: str,
    path: Path,
    artifact: Any,
    relative_to: Path,
    review_threshold: float,
) -> dict[str, Any]:
    if name not in HEAD_POOLINGS:
        raise ValueError(f"unsupported head name: {name!r}")
    review = _finite_probability(review_threshold, f"{name} review threshold")
    block = _finite_probability(
        float(artifact.threshold), f"{name} block threshold", allow_zero=False
    )
    if review >= block:
        raise ValueError(f"{name} review threshold must be below block threshold")
    resolved = path.resolve()
    try:
        reported_path = resolved.relative_to(relative_to.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{name} artifact must be inside the pair directory") from exc
    return {
        "name": name,
        "path": reported_path,
        "sha256": sha256_file(resolved),
        "format": "AEGIS DetectorArtifact NPZ version 1",
        "pooling": artifact.pooling,
        "feature_dim": int(artifact.feature_dim),
        "block_threshold": block,
        "review_threshold": review,
        "source": artifact.source,
    }


def build_pair_manifest(
    *,
    target: str,
    image_path: Path,
    image_artifact: Any,
    image_review_threshold: float,
    text_path: Path,
    text_artifact: Any,
    text_review_threshold: float,
    output_directory: Path,
    training_identity_sha256_value: str,
    corpus: Mapping[str, Any],
    development_qualification_evidence: Mapping[str, Any] | None = None,
    qualification_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    training_digest = _require_sha256(
        training_identity_sha256_value, "training identity sha256"
    )
    image_provenance = artifact_provenance(image_artifact)
    text_provenance = artifact_provenance(text_artifact)
    if image_provenance != text_provenance:
        raise ValueError("image and text artifact provenance does not match")
    base_dim = int(image_artifact.feature_dim)
    if base_dim <= 0 or int(text_artifact.feature_dim) != base_dim:
        raise ValueError("image and text artifact dimensions do not match")
    if image_artifact.pooling != "image_tokens":
        raise ValueError("image artifact must use image_tokens pooling")
    if text_artifact.pooling != "text_tokens":
        raise ValueError("text artifact must use text_tokens pooling")
    entries = {
        "image": artifact_entry(
            name="image",
            path=image_path,
            artifact=image_artifact,
            relative_to=output_directory,
            review_threshold=image_review_threshold,
        ),
        "text": artifact_entry(
            name="text",
            path=text_path,
            artifact=text_artifact,
            relative_to=output_directory,
            review_threshold=text_review_threshold,
        ),
    }
    runtime_identity = runtime_detector_identity_sha256(entries)
    manifest: dict[str, Any] = {
        "schema_version": PAIR_MANIFEST_SCHEMA_VERSION,
        "kind": PAIR_KIND,
        "composition": PAIR_COMPOSITION,
        "target": target,
        "training_identity_sha256": training_digest,
        "corpus": dict(corpus),
        "shared_provenance": {**image_provenance, "base_feature_dim": base_dim},
        "cache_representation": {
            "pooling": CACHE_POOLING,
            "feature_dim": base_dim * 2,
            "ordering": list(CACHE_ORDERING),
        },
        "development_qualification_evidence": (
            None
            if development_qualification_evidence is None
            else dict(development_qualification_evidence)
        ),
        "qualification_evidence": (
            None if qualification_evidence is None else dict(qualification_evidence)
        ),
        "artifacts": entries,
        "runtime_detector_identity_sha256": runtime_identity,
    }
    manifest["pair_identity_sha256"] = pair_identity_sha256(manifest)
    return manifest


def _resolve_manifest_artifact(manifest_path: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} artifact path is missing")
    root = manifest_path.parent.resolve()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} artifact path escapes the pair directory") from exc
    return candidate


def validate_pair_manifest_payload(
    manifest: Mapping[str, Any],
    *,
    expected_target: str | None = None,
    expected_base_feature_dim: int | None = None,
) -> None:
    for name, expected in (
        ("schema_version", PAIR_MANIFEST_SCHEMA_VERSION),
        ("kind", PAIR_KIND),
        ("composition", PAIR_COMPOSITION),
    ):
        if manifest.get(name) != expected:
            raise ValueError(f"unexpected pair manifest {name}: {manifest.get(name)!r}")
    target = manifest.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError("pair manifest target is missing")
    if expected_target is not None and target != expected_target:
        raise ValueError(f"pair target mismatch: {target!r} != {expected_target!r}")
    _require_sha256(manifest.get("training_identity_sha256"), "training identity")
    corpus = manifest.get("corpus")
    if not isinstance(corpus, Mapping):
        raise ValueError("pair manifest corpus binding is missing")
    for name in (
        "manifest_sha256",
        "development_metadata_sha256",
        "regression_metadata_sha256",
    ):
        _require_sha256(corpus.get(name), f"corpus {name}")
    shared = manifest.get("shared_provenance")
    if not isinstance(shared, Mapping):
        raise ValueError("pair manifest shared provenance is missing")
    for name in (
        "model_family",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "preprocessing_sha256",
    ):
        if not isinstance(shared.get(name), str) or not shared.get(name):
            raise ValueError(f"shared provenance {name} is missing")
    _require_sha256(shared.get("preprocessing_sha256"), "preprocessing sha256")
    try:
        base_dim = int(shared.get("base_feature_dim"))
        layer = int(shared.get("layer"))
    except (TypeError, ValueError) as exc:
        raise ValueError("shared feature dimension/layer must be integers") from exc
    if base_dim <= 0 or layer != -1:
        raise ValueError("pair requires a positive base dimension and layer -1")
    if expected_base_feature_dim is not None and base_dim != expected_base_feature_dim:
        raise ValueError(
            f"pair base dimension mismatch: {base_dim} != {expected_base_feature_dim}"
        )
    cache = manifest.get("cache_representation")
    expected_cache = {
        "pooling": CACHE_POOLING,
        "feature_dim": base_dim * 2,
        "ordering": list(CACHE_ORDERING),
    }
    if cache != expected_cache:
        raise ValueError("pair cache representation is not text-then-image fused")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(HEAD_POOLINGS):
        raise ValueError("pair manifest requires exactly image and text artifacts")
    for name, expected_pooling in HEAD_POOLINGS.items():
        entry = artifacts[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"{name} artifact entry must be an object")
        if entry.get("name") != name or entry.get("pooling") != expected_pooling:
            raise ValueError(f"{name} artifact identity/pooling mismatch")
        if entry.get("feature_dim") != base_dim:
            raise ValueError(f"{name} artifact feature dimension mismatch")
        _require_sha256(entry.get("sha256"), f"{name} artifact sha256")
        block = _finite_probability(
            entry.get("block_threshold"),
            f"{name} block threshold",
            allow_zero=False,
        )
        review = _finite_probability(
            entry.get("review_threshold"), f"{name} review threshold"
        )
        if review >= block:
            raise ValueError(f"{name} review threshold must be below block threshold")
    runtime_digest = runtime_detector_identity_sha256(artifacts)
    if manifest.get("runtime_detector_identity_sha256") != runtime_digest:
        raise ValueError("runtime detector identity does not match pair heads")
    qualification = manifest.get("qualification_evidence")
    development_evidence = manifest.get("development_qualification_evidence")
    if development_evidence is not None:
        if not isinstance(development_evidence, Mapping):
            raise ValueError("pair development qualification evidence must be an object")
        if development_evidence.get("schema_version") != 1:
            raise ValueError("unsupported pair development qualification evidence schema")
        snapshot = development_evidence.get("feature_snapshot")
        predictions = development_evidence.get("predictions")
        evidence_corpus = development_evidence.get("corpus")
        evidence_artifacts = development_evidence.get("artifact_sha256")
        if not isinstance(snapshot, Mapping):
            raise ValueError("pair development evidence has no feature snapshot")
        _require_sha256(snapshot.get("sha256"), "development feature snapshot sha256")
        if not isinstance(predictions, Mapping) or set(predictions) != {
            "validation",
            "internal_test",
            "prior_regression",
        }:
            raise ValueError("pair development prediction bindings are incomplete")
        for panel, entry in predictions.items():
            if not isinstance(entry, Mapping):
                raise ValueError(f"pair development prediction {panel} is invalid")
            _require_sha256(entry.get("sha256"), f"development prediction {panel} sha256")
        if not isinstance(evidence_corpus, Mapping) or dict(evidence_corpus) != {
            "manifest_sha256": corpus.get("manifest_sha256"),
            "development_metadata_sha256": corpus.get("development_metadata_sha256"),
            "regression_metadata_sha256": corpus.get("regression_metadata_sha256"),
        }:
            raise ValueError("pair development evidence corpus binding differs")
        if not isinstance(evidence_artifacts, Mapping) or dict(evidence_artifacts) != {
            "image": artifacts["image"]["sha256"],
            "text": artifacts["text"]["sha256"],
        }:
            raise ValueError("pair development evidence artifact binding differs")
    if qualification is not None:
        if not isinstance(qualification, Mapping):
            raise ValueError("pair qualification evidence must be an object or null")
        if (
            qualification.get("schema_version") != ONE_SHOT_SCHEMA_VERSION
            or qualification.get("kind") != ONE_SHOT_KIND
            or qualification.get("passed") is not True
        ):
            raise ValueError("pair qualification evidence is not a passing one-shot result")
        files = qualification.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("pair qualification evidence has no file bindings")
        required_files = {
            "attempt_started",
            "development_manifest",
            "evaluator",
            "protocol",
            "results_json",
            "results_csv",
            "evaluation_manifest",
            "validator",
            "validation_report",
            "validation_manifest",
            *(file_key for file_key, _ in ONE_SHOT_METADATA_FILES.values()),
        }
        source_format = qualification.get("source_format")
        if source_format == "generic_dual_evaluator_compat_v1":
            if qualification.get("qualification_execution") != "atomic_extract_evaluate":
                raise ValueError("generic qualification was not atomic extract/evaluate")
            required_files.update({
                "extractor", "declaration_manifest", "implementation_manifest",
                "input_snapshot_manifest", "model_snapshot_manifest",
                "runtime_environment", "strict_corpus_validation", "subject_claim",
                "ordered_metadata", "cache_materialization",
                "generic_results_json", "generic_results_csv",
                "generic_validation_report", "feature_bundle", "feature_manifest",
                "aligned_source_metadata", "training_summary",
                "source_pair_manifest",
            })
            if (not any(str(name).startswith("implementation_snapshot::") for name in files)
                    or not any(str(name).startswith("input_snapshot::") for name in files)):
                raise ValueError("generic qualification snapshot tree bindings are incomplete")
            cache_files = {
                str(name)
                for name in files
                if str(name).startswith("cache::")
            }
            if len(cache_files) != ONE_SHOT_CACHE_STATISTICS["hits"]:
                raise ValueError("generic qualification cache file bindings are incomplete")
            for name in cache_files:
                logical = name.removeprefix("cache::")
                parts = logical.split("/")
                if len(parts) != 2:
                    raise ValueError("generic qualification cache binding name is invalid")
                panel, sample_id = parts
                qualification_cache_locator("0" * 64, panel, sample_id)
                expected_path = f"qualification_evidence/cache/{panel}/{sample_id}.npz"
                if str(files[name].get("path", "")).replace("\\", "/") != expected_path:
                    raise ValueError("generic qualification cache binding path differs")
            required_files.update(
                name for name in files
                if str(name).startswith("implementation_snapshot::")
                or str(name).startswith("input_snapshot::")
                or str(name).startswith("cache::")
            )
            details = qualification.get("generic_derived_validation")
            if (not isinstance(details, Mapping)
                    or details.get("source_format") != source_format):
                raise ValueError("generic qualification independent validation is missing")
            _require_sha256(
                qualification.get("protocol_qualification_subject_sha256"),
                "generic protocol qualification subject",
            )
        elif source_format is not None:
            raise ValueError("unsupported qualification source format")
        if set(files) != required_files:
            raise ValueError("pair qualification evidence file set is incomplete")
        for name, entry in files.items():
            if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
                raise ValueError(f"qualification evidence entry {name!r} is invalid")
            _require_sha256(entry.get("sha256"), f"qualification {name} sha256")
        for name in (
            "image_artifact_sha256",
            "text_artifact_sha256",
            "corpus_manifest_sha256",
            "cache_entry_tree_sha256",
            "runtime_detector_identity_sha256",
            "qualification_subject_sha256",
        ):
            _require_sha256(qualification.get(name), f"qualification {name}")
        if qualification.get("image_artifact_sha256") != artifacts["image"]["sha256"]:
            raise ValueError("qualification evidence binds a different image artifact")
        if qualification.get("text_artifact_sha256") != artifacts["text"]["sha256"]:
            raise ValueError("qualification evidence binds a different text artifact")
        if qualification.get("corpus_manifest_sha256") != corpus.get("manifest_sha256"):
            raise ValueError("qualification evidence binds a different corpus manifest")
        if qualification.get("rows") != 50 or qualification.get("unique_sample_ids") != 50:
            raise ValueError("qualification evidence must bind 50 unique frozen rows")
        if qualification.get("validator_cache") != ONE_SHOT_CACHE_STATISTICS:
            raise ValueError("qualification evidence cache contract did not pass exactly")
        if qualification.get("target") != target:
            raise ValueError("qualification evidence binds a different target")
        if qualification.get("runtime_detector_identity_sha256") != runtime_digest:
            raise ValueError("qualification evidence binds a different runtime identity")
        subject_digest = qualification_subject_sha256(
            target=target,
            corpus_manifest_sha256=str(corpus["manifest_sha256"]),
            artifacts=artifacts,
            shared_provenance={
                key: shared[key]
                for key in (
                    "model_family",
                    "model_id",
                    "model_revision",
                    "tokenizer_revision",
                    "preprocessing_sha256",
                    "layer",
                )
            },
            runtime_detector_identity_sha256_value=runtime_digest,
        )
        if qualification.get("qualification_subject_sha256") != subject_digest:
            raise ValueError("qualification evidence binds a different detector pair")
        if qualification.get("gate_checks") != {
            name: True for name in sorted(ONE_SHOT_GATE_NAMES)
        }:
            raise ValueError("qualification evidence does not bind every frozen gate")
        for name in ("protocol_id", "evaluation_id", "validation_id"):
            if not isinstance(qualification.get(name), str) or not qualification[name]:
                raise ValueError(f"qualification evidence {name} is missing")
        if qualification.get("evaluation_id") != qualification.get("protocol_id"):
            raise ValueError("qualification evaluation/protocol identities differ")
        if qualification.get("promotion_authorized_by_evidence") is not False:
            raise ValueError("qualification evidence must not authorize promotion")
        declared = qualification.get("declared_evidence_sha256")
        if not isinstance(declared, Mapping) or set(declared) != required_files:
            raise ValueError("qualification declared-evidence hash set is incomplete")
        for name in required_files:
            _require_sha256(declared.get(name), f"qualification declared {name} sha256")
            if declared[name] != files[name]["sha256"]:
                raise ValueError(f"qualification declared/file hash differs for {name}")
        frozen_hashes = qualification.get("frozen_metadata_sha256")
        if not isinstance(frozen_hashes, Mapping) or set(frozen_hashes) != {
            "regression",
            "text_led",
            "external_benign",
        }:
            raise ValueError("qualification frozen metadata bindings are incomplete")
        for name, value in frozen_hashes.items():
            _require_sha256(value, f"qualification frozen metadata {name}")
            file_key, _ = ONE_SHOT_METADATA_FILES[name]
            if value != files[file_key]["sha256"]:
                raise ValueError(
                    f"qualification frozen metadata/file hash differs for {name}"
                )
    rich_digest = pair_identity_sha256(manifest)
    if manifest.get("pair_identity_sha256") != rich_digest:
        raise ValueError("pair identity does not match its canonical payload")


def build_one_shot_qualification_binding(
    *,
    evidence_directory: Path,
    pair_directory: Path,
    image_artifact_sha256: str,
    text_artifact_sha256: str,
    corpus_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate and bind the already-consumed one-shot dual evaluation evidence."""

    evidence_root = evidence_directory.expanduser().resolve()
    pair_root = pair_directory.expanduser().resolve()
    development_manifest = evidence_root.parent / "development_manifest.json"
    if not development_manifest.is_file():
        development_manifest = evidence_root.parent.parent / "output_manifest.json"
    paths = {
        "attempt_started": evidence_root / "attempt_started.json",
        "development_manifest": development_manifest,
        "evaluator": evidence_root.parent / "evaluate_once.py",
        "protocol": evidence_root.parent / "protocol.json",
        "results_json": evidence_root / "dual_head_frozen_results.json",
        "results_csv": evidence_root / "dual_head_frozen_results.csv",
        "evaluation_manifest": evidence_root / "evaluation_manifest.json",
        "validator": evidence_root.parent / "validate_once.py",
        "validation_report": evidence_root / "validation_report.json",
        "validation_manifest": evidence_root / "validation_manifest.json",
        **{
            file_key: evidence_root.parent / "metadata" / filename
            for file_key, filename in ONE_SHOT_METADATA_FILES.values()
        },
    }
    if not paths["protocol"].is_file():
        raise FileNotFoundError("one-shot qualification protocol is missing")
    protocol_value = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    if not isinstance(protocol_value, dict):
        raise ValueError("one-shot protocol must contain a JSON object")
    generic_package = protocol_value.get("package_format") == (
        "generic_dual_evaluator_compat_v1"
    )
    if generic_package:
        from .bordair_dual_qualification import generic_derived_paths
        paths.update(generic_derived_paths(evidence_root))
        paths["extractor"] = evidence_root.parent / "extract_once.py"
    if not all(path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise FileNotFoundError(f"one-shot qualification evidence is incomplete: {missing}")
    def read_object(name: str) -> dict[str, Any]:
        value = json.loads(paths[name].read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"one-shot {name} must contain a JSON object")
        return value

    protocol = read_object("protocol")
    attempt = read_object("attempt_started")
    results = read_object("results_json")
    evaluation_manifest = read_object("evaluation_manifest")
    report = read_object("validation_report")
    validation_manifest = read_object("validation_manifest")
    protocol_hash = sha256_file(paths["protocol"])
    protocol_id = protocol.get("protocol_id")
    target = protocol.get("target")
    if (
        protocol.get("schema_version") != ONE_SHOT_SCHEMA_VERSION
        or not isinstance(protocol_id, str)
        or not protocol_id
        or not isinstance(target, str)
        or not target
        or protocol.get("status") != "declared_before_frozen_execution"
        or protocol.get("execution_limit") != 1
        or protocol.get("selection_or_tuning_permitted") is not False
    ):
        raise ValueError("one-shot protocol identity or execution contract is invalid")
    if (
        attempt.get("schema_version") != ONE_SHOT_SCHEMA_VERSION
        or attempt.get("protocol_id") != protocol_id
        or attempt.get("protocol_sha256") != protocol_hash
        or attempt.get("execution_ordinal") != 1
        or attempt.get("retries_permitted") is not False
        or attempt.get("evaluator_sha256") != sha256_file(paths["evaluator"])
    ):
        raise ValueError("one-shot attempt marker is not bound to the protocol/evaluator")
    if (
        results.get("schema_version") != ONE_SHOT_SCHEMA_VERSION
        or results.get("evaluation_id") != protocol_id
        or results.get("promotion_authorized") is not False
        or results.get("one_shot_attempt", {}).get("protocol_id") != protocol_id
        or results.get("one_shot_attempt", {}).get("protocol_sha256") != protocol_hash
        or results.get("one_shot_attempt", {}).get("execution_ordinal") != 1
        or results.get("one_shot_attempt", {}).get("retries_permitted") is not False
        or results.get("protocol", {}).get("sha256") != protocol_hash
        or results.get("protocol", {}).get("payload") != protocol
    ):
        raise ValueError("one-shot result envelope is not bound to the declared protocol")

    expected_hashes = {
        "image_artifact_sha256": _require_sha256(
            image_artifact_sha256, "image artifact sha256"
        ),
        "text_artifact_sha256": _require_sha256(
            text_artifact_sha256, "text artifact sha256"
        ),
        "corpus_manifest_sha256": _require_sha256(
            corpus_manifest_sha256, "corpus manifest sha256"
        ),
    }
    raw_artifacts = results.get("artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise ValueError("one-shot results have no artifact contract")
    raw_shared = raw_artifacts.get("shared_provenance")
    if not isinstance(raw_shared, Mapping):
        raise ValueError("one-shot results have no shared artifact provenance")
    shared = {
        name: raw_shared.get(name)
        for name in (
            "model_family",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "preprocessing_sha256",
            "layer",
        )
    }
    normalized_artifacts: dict[str, dict[str, Any]] = {}
    for name, result_name, expected_hash in (
        ("image", "image_head", expected_hashes["image_artifact_sha256"]),
        ("text", "text_head", expected_hashes["text_artifact_sha256"]),
    ):
        raw = raw_artifacts.get(result_name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"one-shot results have no {name} artifact contract")
        block = _finite_probability(
            raw.get("block_threshold"), f"one-shot {name} block threshold", allow_zero=False
        )
        review = _finite_probability(
            raw.get("review_threshold"), f"one-shot {name} review threshold"
        )
        margin = _finite_probability(
            raw.get("uncertainty_margin"), f"one-shot {name} uncertainty margin"
        )
        expected_review = max(0.0, block - margin)
        if review != expected_review:
            raise ValueError(f"one-shot {name} review threshold is not artifact-derived")
        feature_dim = raw.get("feature_dim")
        if (
            raw.get("artifact_version") != 1
            or raw.get("sha256") != expected_hash
            or raw.get("pooling") != HEAD_POOLINGS[name]
            or isinstance(feature_dim, bool)
            or not isinstance(feature_dim, int)
            or feature_dim <= 0
            or any(raw.get(field) != shared[field] for field in shared)
        ):
            raise ValueError(f"one-shot {name} artifact contract differs")
        normalized_artifacts[name] = {
            "sha256": expected_hash,
            "pooling": HEAD_POOLINGS[name],
            "feature_dim": feature_dim,
            "block_threshold": block,
            "review_threshold": review,
        }
    if normalized_artifacts["image"]["feature_dim"] != normalized_artifacts["text"][
        "feature_dim"
    ]:
        raise ValueError("one-shot artifact dimensions differ")
    base_dim = int(normalized_artifacts["image"]["feature_dim"])
    _require_sha256(shared.get("preprocessing_sha256"), "one-shot preprocessing sha256")
    if shared.get("layer") != -1:
        raise ValueError("one-shot artifacts must use layer -1")
    runtime_identity = runtime_detector_identity_sha256(normalized_artifacts)
    subject_digest = qualification_subject_sha256(
        target=target,
        corpus_manifest_sha256=expected_hashes["corpus_manifest_sha256"],
        artifacts=normalized_artifacts,
        shared_provenance=shared,
        runtime_detector_identity_sha256_value=runtime_identity,
    )

    development_hash = sha256_file(paths["development_manifest"])
    development_binding = protocol.get("development_binding")
    corpus_binding = protocol.get("corpus_binding")
    if (
        not isinstance(development_binding, Mapping)
        or development_binding.get("manifest_sha256") != development_hash
        or development_binding.get("image_artifact_sha256")
        != expected_hashes["image_artifact_sha256"]
        or development_binding.get("text_artifact_sha256")
        != expected_hashes["text_artifact_sha256"]
        or not isinstance(corpus_binding, Mapping)
        or corpus_binding.get("manifest_schema") != 4
        or corpus_binding.get("manifest_sha256")
        != expected_hashes["corpus_manifest_sha256"]
    ):
        raise ValueError("one-shot protocol development/corpus binding differs")
    frozen_metadata_hashes = {
        "regression": _require_sha256(
            corpus_binding.get("fixed_metadata_sha256"), "fixed metadata sha256"
        ),
        "text_led": _require_sha256(
            corpus_binding.get("text_led_metadata_sha256"), "text-led metadata sha256"
        ),
        "external_benign": _require_sha256(
            corpus_binding.get("external_benign_metadata_sha256"),
            "external benign metadata sha256",
        ),
    }
    expected_metadata_counts = {
        "regression": 20,
        "text_led": 10,
        "external_benign": 20,
    }
    authoritative_rows: list[dict[str, Any]] = []
    for panel in ("regression", "text_led", "external_benign"):
        file_key, _ = ONE_SHOT_METADATA_FILES[panel]
        metadata_path = paths[file_key]
        if sha256_file(metadata_path) != frozen_metadata_hashes[panel]:
            raise ValueError(f"one-shot authoritative metadata hash differs: {panel}")
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {
                "sample_id",
                "label_id",
                "prompt_text",
                "image_sha256",
            }
            if reader.fieldnames is None or not required_columns.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"one-shot authoritative metadata columns differ: {panel}"
                )
            metadata_rows = list(reader)
        if len(metadata_rows) != expected_metadata_counts[panel]:
            raise ValueError(f"one-shot authoritative metadata count differs: {panel}")
        for metadata_row in metadata_rows:
            sample_id = metadata_row.get("sample_id")
            prompt_text = metadata_row.get("prompt_text")
            if not isinstance(sample_id, str) or not sample_id or not isinstance(
                prompt_text, str
            ):
                raise ValueError(
                    f"one-shot authoritative metadata identity differs: {panel}"
                )
            try:
                label_id = int(metadata_row["label_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"one-shot authoritative metadata label differs: {panel}"
                ) from exc
            if label_id not in {0, 1}:
                raise ValueError(
                    f"one-shot authoritative metadata label differs: {panel}"
                )
            authoritative_rows.append(
                {
                    "panel": panel,
                    "sample_id": sample_id,
                    "label_id": label_id,
                    "image_sha256": _require_sha256(
                        metadata_row.get("image_sha256"),
                        f"one-shot authoritative {panel} image sha256",
                    ),
                    "caller_text_sha256": hashlib.sha256(
                        prompt_text.encode("utf-8")
                    ).hexdigest(),
                }
            )
    protocol_artifact = protocol.get("artifact_contract")
    protocol_cache = protocol.get("cache_contract")
    if (
        not isinstance(protocol_artifact, Mapping)
        or protocol_artifact.get("format") != "AEGIS DetectorArtifact NPZ version 1"
        or protocol_artifact.get("image_pooling") != "image_tokens"
        or protocol_artifact.get("text_pooling") != "text_tokens"
        or protocol_artifact.get("per_head_dimension") != base_dim
        or not isinstance(protocol_cache, Mapping)
        or protocol_cache.get("pooling") != CACHE_POOLING
        or protocol_cache.get("dimension") != base_dim * 2
        or protocol_cache.get("expected_rows") != 50
        or protocol_cache.get("required_statistics") != ONE_SHOT_CACHE_STATISTICS
        or protocol_cache.get("split_order")
        != [f"text_tokens[0:{base_dim}]", f"image_tokens[{base_dim}:{base_dim * 2}]"]
        or results.get("split_contract")
        != {
            "fused_dimension": base_dim * 2,
            "text_slice": [0, base_dim],
            "image_slice": [base_dim, base_dim * 2],
        }
        or results.get("action_contract") != protocol.get("action_contract")
    ):
        raise ValueError("one-shot artifact/cache/action contract differs")

    rows = results.get("results")
    if (
        not isinstance(rows, list)
        or len(rows) != 50
        or not all(isinstance(row, Mapping) for row in rows)
    ):
        raise ValueError("one-shot results must contain exactly 50 rows")
    if len({row.get("sample_id") for row in rows}) != 50:
        raise ValueError("one-shot results must contain 50 unique sample IDs")
    if [row.get("sequence") for row in rows] != list(range(1, 51)):
        raise ValueError("one-shot result sequence must be exactly 1 through 50")
    for sequence, (row, authoritative) in enumerate(
        zip(rows, authoritative_rows), start=1
    ):
        if any(row.get(name) != authoritative[name] for name in authoritative):
            raise ValueError(
                "one-shot result row differs from authoritative metadata: "
                f"sequence {sequence}"
            )
    expected_panel_counts = {
        "regression": {"rows": 20, "malicious": 10, "benign": 10},
        "text_led": {"rows": 10, "malicious": 10, "benign": 0},
        "external_benign": {"rows": 20, "malicious": 0, "benign": 20},
    }
    recomputed_panels: dict[str, dict[str, Any]] = {}
    for panel, counts in expected_panel_counts.items():
        selected = [row for row in rows if isinstance(row, Mapping) and row.get("panel") == panel]
        action_counts = {action: 0 for action in ACTION_ORDER}
        image_counts = {action: 0 for action in ACTION_ORDER}
        text_counts = {action: 0 for action in ACTION_ORDER}
        malicious_blocked = 0
        benign_blocked = 0
        for row in selected:
            try:
                sequence = int(row["sequence"])
                label = int(row["label_id"])
                image_score = float(row["image_head_score"])
                text_score = float(row["text_head_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("one-shot result row has invalid scalar fields") from exc
            if sequence < 1 or sequence > 50 or not math.isfinite(image_score) or not math.isfinite(text_score):
                raise ValueError("one-shot result row has invalid sequence/score")
            image_action = head_action(
                image_score,
                block_threshold=float(normalized_artifacts["image"]["block_threshold"]),
                review_threshold=float(normalized_artifacts["image"]["review_threshold"]),
            )
            text_action = head_action(
                text_score,
                block_threshold=float(normalized_artifacts["text"]["block_threshold"]),
                review_threshold=float(normalized_artifacts["text"]["review_threshold"]),
            )
            combined = combine_actions(image_action, text_action)
            expected_action = "block" if label == 1 else "allow"
            exact = {
                "image_head_block_threshold": normalized_artifacts["image"]["block_threshold"],
                "image_head_review_threshold": normalized_artifacts["image"]["review_threshold"],
                "text_head_block_threshold": normalized_artifacts["text"]["block_threshold"],
                "text_head_review_threshold": normalized_artifacts["text"]["review_threshold"],
                "image_head_action": image_action,
                "text_head_action": text_action,
                "combined_action": combined,
                "expected_action": expected_action,
                "accepted": combined == expected_action,
            }
            if any(row.get(name) != value for name, value in exact.items()):
                raise ValueError(f"one-shot row action/threshold evidence differs: {row.get('sample_id')}")
            _require_sha256(row.get("cache_sha256"), "one-shot row cache sha256")
            action_counts[combined] += 1
            image_counts[image_action] += 1
            text_counts[text_action] += 1
            malicious_blocked += int(label == 1 and combined == "block")
            benign_blocked += int(label == 0 and combined == "block")
        malicious_rows = sum(int(row["label_id"]) == 1 for row in selected)
        benign_rows = sum(int(row["label_id"]) == 0 for row in selected)
        if (
            len(selected) != counts["rows"]
            or malicious_rows != counts["malicious"]
            or benign_rows != counts["benign"]
        ):
            raise ValueError(f"one-shot panel count differs: {panel}")
        recomputed_panels[panel] = {
            "rows": len(selected),
            "malicious_rows": malicious_rows,
            "benign_rows": benign_rows,
            "malicious_blocked": malicious_blocked,
            "benign_blocked": benign_blocked,
            "combined_action_counts": action_counts,
            "image_head_action_counts": image_counts,
            "text_head_action_counts": text_counts,
        }
    gate_checks = {
        "external_benign_exact_counts": recomputed_panels["external_benign"]["benign_rows"] == 20,
        "external_benign_fewer_than_3_blocked": recomputed_panels["external_benign"]["benign_blocked"] < 3,
        "fixed_benign_fewer_than_3_blocked": recomputed_panels["regression"]["benign_blocked"] < 3,
        "fixed_exact_counts": recomputed_panels["regression"]["malicious_rows"] == 10 and recomputed_panels["regression"]["benign_rows"] == 10,
        "fixed_malicious_10_of_10_blocked": recomputed_panels["regression"]["malicious_blocked"] == 10,
        "text_led_exact_counts": recomputed_panels["text_led"]["malicious_rows"] == 10,
        "text_led_malicious_10_of_10_blocked": recomputed_panels["text_led"]["malicious_blocked"] == 10,
    }
    expected_gates = {"checks": gate_checks, "passed": all(gate_checks.values())}
    if set(gate_checks) != ONE_SHOT_GATE_NAMES or results.get("gates") != expected_gates or results.get("panels") != recomputed_panels:
        raise ValueError("one-shot gates/panel summaries do not recompute")

    cache = results.get("cache")
    cache_report = cache.get("report") if isinstance(cache, Mapping) else None
    if (
        not isinstance(cache_report, Mapping)
        or cache_report.get("enabled") is not True
        or cache_report.get("schema_version") != 1
        or cache_report.get("expected_rows") != 50
        or cache_report.get("rebuild_requested") is not False
        or cache_report.get("write_error_details") != []
        or {name: cache_report.get(name) for name in ONE_SHOT_CACHE_STATISTICS}
        != ONE_SHOT_CACHE_STATISTICS
        or cache.get("delegate_object_calls") != 0
    ):
        raise ValueError("one-shot evaluator cache contract did not pass exactly")
    entries = cache.get("entries")
    if not isinstance(entries, list) or len(entries) != 50:
        raise ValueError("one-shot cache evidence must contain 50 entries")
    digest_lines: list[str] = []
    for row, entry in zip(rows, entries):
        if not isinstance(entry, Mapping):
            raise ValueError("one-shot cache entry is invalid")
        digest = _require_sha256(entry.get("sha256"), "one-shot cache entry sha256")
        byte_count = entry.get("bytes")
        path_value = entry.get("path")
        if (
            entry.get("panel") != row.get("panel")
            or entry.get("sample_id") != row.get("sample_id")
            or entry.get("sha256") != row.get("cache_sha256")
            or entry.get("path") != row.get("cache_path")
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not isinstance(path_value, str)
        ):
            raise ValueError("one-shot cache entry differs from its result row")
        digest_lines.append(f"{path_value}|{byte_count}|{digest}")
    cache_tree = hashlib.sha256(("\n".join(digest_lines) + "\n").encode("utf-8")).hexdigest()
    if cache.get("entry_tree_sha256") != cache_tree:
        raise ValueError("one-shot cache entry tree does not recompute")

    output_hashes = {
        "dual_head_frozen_results.json": sha256_file(paths["results_json"]),
        "dual_head_frozen_results.csv": sha256_file(paths["results_csv"]),
    }
    if (
        evaluation_manifest.get("schema_version") != ONE_SHOT_SCHEMA_VERSION
        or evaluation_manifest.get("evaluation_id") != protocol_id
        or evaluation_manifest.get("row_count") != 50
        or evaluation_manifest.get("corpus_manifest_sha256") != expected_hashes["corpus_manifest_sha256"]
        or evaluation_manifest.get("cache_entry_tree_sha256") != cache_tree
        or evaluation_manifest.get("artifacts")
        != {
            "image_head_sha256": expected_hashes["image_artifact_sha256"],
            "text_head_sha256": expected_hashes["text_artifact_sha256"],
        }
        or evaluation_manifest.get("development_manifest", {}).get("sha256") != development_hash
        or evaluation_manifest.get("protocol", {}).get("sha256") != protocol_hash
        or evaluation_manifest.get("attempt", {}).get("sha256") != sha256_file(paths["attempt_started"])
        or evaluation_manifest.get("evaluator", {}).get("sha256") != sha256_file(paths["evaluator"])
        or evaluation_manifest.get("validator", {}).get("sha256") != sha256_file(paths["validator"])
        or evaluation_manifest.get("panel_metadata_sha256") != frozen_metadata_hashes
    ):
        raise ValueError("one-shot evaluation manifest binding differs")
    manifest_outputs = evaluation_manifest.get("outputs")
    if not isinstance(manifest_outputs, Mapping):
        raise ValueError("one-shot evaluation manifest has no output bindings")
    for filename, path_name in (
        ("dual_head_frozen_results.json", "results_json"),
        ("dual_head_frozen_results.csv", "results_csv"),
    ):
        entry = manifest_outputs.get(filename)
        if (
            not isinstance(entry, Mapping)
            or entry.get("sha256") != output_hashes[filename]
            or entry.get("bytes") != paths[path_name].stat().st_size
        ):
            raise ValueError(f"one-shot evaluation manifest does not bind {filename}")

    if (
        report.get("schema_version") != ONE_SHOT_SCHEMA_VERSION
        or report.get("ok") is not True
        or report.get("promotion_authorized") is not False
        or not isinstance(report.get("validation_id"), str)
        or not report.get("validation_id")
        or report.get("gates") != expected_gates
        or report.get("panels") != recomputed_panels
        or report.get("evaluation_manifest", {}).get("sha256")
        != sha256_file(paths["evaluation_manifest"])
    ):
        raise ValueError("one-shot validation report did not independently pass")
    gates = report.get("gates")
    recomputation = report.get("recomputation")
    evidence = report.get("evidence")
    if (
        not isinstance(gates, dict)
        or gates.get("passed") is not True
        or not isinstance(recomputation, dict)
        or recomputation.get("all_per_head_scores_and_actions_recomputed") is not True
        or recomputation.get("all_combined_actions_recomputed") is not True
        or recomputation.get("csv_json_exact_copy_validated") is not True
        or not isinstance(evidence, dict)
    ):
        raise ValueError("one-shot independent validation evidence is incomplete")
    for name, value in expected_hashes.items():
        if evidence.get(name) != value:
            raise ValueError(f"one-shot validation evidence differs for {name}")
    cache_tree = _require_sha256(
        evidence.get("cache_entry_tree_sha256"), "cache entry tree sha256"
    )
    if evidence.get("development_manifest_sha256") != development_hash:
        raise ValueError("validation report does not bind the development manifest")
    for report_name, path_name in (
        ("results_json", "results_json"),
        ("results_csv", "results_csv"),
    ):
        reported_file = evidence.get(report_name)
        if not isinstance(reported_file, dict) or reported_file.get(
            "sha256"
        ) != sha256_file(paths[path_name]):
            raise ValueError(f"validation report does not bind {report_name}")
    if evidence.get("protocol_sha256") != sha256_file(paths["protocol"]):
        raise ValueError("validation report does not bind the protocol")
    validator_cache = recomputation.get("validator_cache")
    if (
        not isinstance(validator_cache, dict)
        or {name: validator_cache.get(name) for name in ONE_SHOT_CACHE_STATISTICS}
        != ONE_SHOT_CACHE_STATISTICS
        or validator_cache.get("enabled") is not True
        or validator_cache.get("schema_version") != 1
        or validator_cache.get("expected_rows") != 50
        or validator_cache.get("rebuild_requested") is not False
        or validator_cache.get("write_error_details") != []
    ):
        raise ValueError("one-shot validator cache contract did not pass exactly")
    if recomputation.get("rows") != 50 or recomputation.get("unique_sample_ids") != 50:
        raise ValueError("one-shot validator did not recompute 50 unique rows")
    validation_report_binding = validation_manifest.get("validation_report")
    if (
        validation_manifest.get("schema_version") != ONE_SHOT_SCHEMA_VERSION
        or validation_manifest.get("promotion_authorized") is not False
        or not isinstance(validation_report_binding, dict)
        or validation_report_binding.get("sha256")
        != sha256_file(paths["validation_report"])
        or validation_report_binding.get("bytes") != paths["validation_report"].stat().st_size
        or validation_manifest.get("evaluation_manifest", {}).get("sha256")
        != sha256_file(paths["evaluation_manifest"])
        or validation_manifest.get("validator", {}).get("sha256")
        != sha256_file(paths["validator"])
    ):
        raise ValueError("one-shot validation manifest is invalid")
    generic_details = None
    if generic_package:
        from .bordair_dual_qualification import validate_generic_derived_evidence
        generic_details = validate_generic_derived_evidence(
            evidence_root=evidence_root,
            protocol=protocol,
            results=results,
            report=report,
            evaluation_manifest=evaluation_manifest,
            validation_manifest=validation_manifest,
            compatibility_rows=rows,
            compatibility_panels=recomputed_panels,
            compatibility_gates=expected_gates,
            expected_image_sha256=expected_hashes["image_artifact_sha256"],
            expected_text_sha256=expected_hashes["text_artifact_sha256"],
            expected_corpus_sha256=expected_hashes["corpus_manifest_sha256"],
            expected_runtime_identity=runtime_identity,
        )
    file_bindings: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        try:
            reported_path = path.relative_to(pair_root).as_posix()
        except ValueError as exc:
            raise ValueError("qualification evidence must be inside the pair tree") from exc
        file_bindings[name] = {"path": reported_path, "sha256": sha256_file(path)}
    declared_hashes = {
        "attempt_started": sha256_file(paths["attempt_started"]),
        "development_manifest": development_hash,
        "evaluator": sha256_file(paths["evaluator"]),
        "evaluation_manifest": sha256_file(paths["evaluation_manifest"]),
        "protocol": protocol_hash,
        "results_csv": output_hashes["dual_head_frozen_results.csv"],
        "results_json": output_hashes["dual_head_frozen_results.json"],
        "validation_manifest": sha256_file(paths["validation_manifest"]),
        "validation_report": sha256_file(paths["validation_report"]),
        "validator": sha256_file(paths["validator"]),
        **{
            file_key: sha256_file(paths[file_key])
            for file_key, _ in ONE_SHOT_METADATA_FILES.values()
        },
    }
    if generic_package:
        declared_hashes = {name: sha256_file(path) for name, path in paths.items()}
    binding = {
        "schema_version": ONE_SHOT_SCHEMA_VERSION,
        "kind": ONE_SHOT_KIND,
        "passed": True,
        "target": target,
        "protocol_id": protocol_id,
        "evaluation_id": str(results["evaluation_id"]),
        "validation_id": str(report["validation_id"]),
        **expected_hashes,
        "runtime_detector_identity_sha256": runtime_identity,
        "qualification_subject_sha256": subject_digest,
        "cache_entry_tree_sha256": cache_tree,
        "rows": int(recomputation.get("rows", 0)),
        "unique_sample_ids": int(recomputation.get("unique_sample_ids", 0)),
        "validator_cache": dict(ONE_SHOT_CACHE_STATISTICS),
        "files": file_bindings,
        "gate_checks": {name: True for name in sorted(ONE_SHOT_GATE_NAMES)},
        "frozen_metadata_sha256": frozen_metadata_hashes,
        "declared_evidence_sha256": declared_hashes,
        "promotion_authorized_by_evidence": False,
    }
    if generic_package:
        binding.update({
            "source_format": "generic_dual_evaluator_compat_v1",
            "qualification_execution": "atomic_extract_evaluate",
            "protocol_qualification_subject_sha256": protocol[
                "qualification_subject_sha256"
            ],
            "generic_derived_validation": generic_details,
        })
    return binding


def validate_bound_qualification_files(pair: LoadedDetectorPair) -> dict[str, Any] | None:
    qualification = pair.manifest.get("qualification_evidence")
    if qualification is None:
        return None
    validate_pair_manifest_payload(pair.manifest)
    assert isinstance(qualification, dict)
    resolved_files: dict[str, Path] = {}
    for name, entry in qualification["files"].items():
        path = _resolve_manifest_artifact(
            pair.manifest_path, entry["path"], f"qualification {name}"
        )
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"bound qualification evidence changed: {name}")
        resolved_files[name] = path
    rebuilt = build_one_shot_qualification_binding(
        evidence_directory=resolved_files["results_json"].parent,
        pair_directory=pair.manifest_path.parent,
        image_artifact_sha256=pair.manifest["artifacts"]["image"]["sha256"],
        text_artifact_sha256=pair.manifest["artifacts"]["text"]["sha256"],
        corpus_manifest_sha256=pair.manifest["corpus"]["manifest_sha256"],
    )
    if rebuilt != qualification:
        raise ValueError("bound qualification evidence does not revalidate exactly")
    return qualification


def load_detector_pair(
    manifest_path: Path,
    *,
    expected_target: str | None = None,
    expected_base_feature_dim: int | None = None,
    image_override: Path | None = None,
    text_override: Path | None = None,
) -> LoadedDetectorPair:
    from AEGIS.detector_artifact import load_detector_artifact

    resolved_manifest = manifest_path.expanduser().resolve()
    raw = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pair manifest must contain a JSON object")
    validate_pair_manifest_payload(
        raw,
        expected_target=expected_target,
        expected_base_feature_dim=expected_base_feature_dim,
    )
    entries = raw["artifacts"]
    declared_image = _resolve_manifest_artifact(
        resolved_manifest, entries["image"].get("path"), "image"
    )
    declared_text = _resolve_manifest_artifact(
        resolved_manifest, entries["text"].get("path"), "text"
    )
    image_path = declared_image if image_override is None else image_override.resolve()
    text_path = declared_text if text_override is None else text_override.resolve()
    for name, selected, declared in (
        ("image", image_path, declared_image),
        ("text", text_path, declared_text),
    ):
        if not selected.is_file():
            raise FileNotFoundError(f"{name} detector artifact is missing: {selected}")
        digest = sha256_file(selected)
        if digest != entries[name]["sha256"]:
            raise ValueError(f"{name} detector artifact SHA-256 mismatch")
        if sha256_file(declared) != digest:
            raise ValueError(f"{name} artifact override is not byte-identical")
    image_artifact = load_detector_artifact(image_path)
    text_artifact = load_detector_artifact(text_path)
    shared = raw["shared_provenance"]
    base_dim = int(shared["base_feature_dim"])
    for name, artifact in (
        ("image", image_artifact),
        ("text", text_artifact),
    ):
        entry = entries[name]
        if artifact_provenance(artifact) != {
            key: shared[key]
            for key in (
                "model_family",
                "model_id",
                "model_revision",
                "tokenizer_revision",
                "preprocessing_sha256",
                "layer",
            )
        }:
            raise ValueError(f"{name} artifact provenance differs from pair manifest")
        if artifact.pooling != HEAD_POOLINGS[name]:
            raise ValueError(f"{name} artifact pooling mismatch")
        if artifact.feature_dim != base_dim:
            raise ValueError(f"{name} artifact dimension mismatch")
        if float(artifact.threshold) != float(entry["block_threshold"]):
            raise ValueError(f"{name} artifact threshold mismatch")
        if artifact.source != entry.get("source"):
            raise ValueError(f"{name} artifact source mismatch")
    if artifact_provenance(image_artifact) != artifact_provenance(text_artifact):
        raise ValueError("image and text artifacts do not share provenance")
    pair = LoadedDetectorPair(
        manifest=raw,
        manifest_path=resolved_manifest,
        image_artifact=image_artifact,
        text_artifact=text_artifact,
        paths=PairArtifactPaths(image=image_path, text=text_path),
    )
    validate_bound_qualification_files(pair)
    return pair


def split_fused_features(
    fused: np.ndarray, base_feature_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(fused, dtype=np.float64)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != base_feature_dim * 2:
        raise ValueError(
            "fused feature vector must be text-then-image with dimension "
            f"{base_feature_dim * 2}; got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("fused features contain non-finite values")
    text = values[:, :base_feature_dim]
    image = values[:, base_feature_dim:]
    if one_dimensional:
        return text.reshape(-1), image.reshape(-1)
    return text, image


def head_action(score: float, *, block_threshold: float, review_threshold: float) -> str:
    value = float(score)
    if not math.isfinite(value):
        raise ValueError("head score must be finite")
    if value >= block_threshold:
        return "block"
    if value >= review_threshold:
        return "review"
    return "allow"


def combine_actions(*actions: str) -> str:
    if not actions:
        raise ValueError("at least one head action is required")
    if any(action not in ACTION_ORDER for action in actions):
        raise ValueError(f"unsupported head action: {actions!r}")
    return max(actions, key=ACTION_ORDER.__getitem__)


def score_pair(pair: LoadedDetectorPair, fused_features: np.ndarray) -> dict[str, Any]:
    text, image = split_fused_features(fused_features, pair.base_feature_dim)
    if text.ndim == 1:
        text = text.reshape(1, -1)
        image = image.reshape(1, -1)
    image_scores = pair.image_artifact.score(image)
    text_scores = pair.text_artifact.score(text)
    entries = pair.manifest["artifacts"]
    image_actions = [
        head_action(
            float(score),
            block_threshold=float(pair.image_artifact.threshold),
            review_threshold=float(entries["image"]["review_threshold"]),
        )
        for score in image_scores
    ]
    text_actions = [
        head_action(
            float(score),
            block_threshold=float(pair.text_artifact.threshold),
            review_threshold=float(entries["text"]["review_threshold"]),
        )
        for score in text_scores
    ]
    return {
        "image_scores": image_scores,
        "text_scores": text_scores,
        "image_actions": image_actions,
        "text_actions": text_actions,
        "combined_actions": [
            combine_actions(image_action, text_action)
            for image_action, text_action in zip(image_actions, text_actions)
        ],
    }


def logit(values: np.ndarray | float) -> np.ndarray:
    epsilon = np.finfo(np.float64).eps
    array = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(array) - np.log1p(-array)


def _sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -40.0, 40.0)))


def _logit_midpoint(lower: float, upper: float) -> float:
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(f"invalid threshold endpoints: {lower}, {upper}")
    return float(_sigmoid((float(logit(lower)) + float(logit(upper))) / 2.0))


def calibrate_channel_threshold(
    benign_scores: np.ndarray,
    relevant_malicious_scores: np.ndarray,
    false_positive_budget: int,
) -> dict[str, Any]:
    benign = np.asarray(benign_scores, dtype=np.float64)
    malicious = np.asarray(relevant_malicious_scores, dtype=np.float64)
    if benign.ndim != 1 or malicious.ndim != 1 or not len(benign) or not len(malicious):
        raise ValueError("threshold calibration requires non-empty 1D class scores")
    if false_positive_budget < 0 or false_positive_budget >= len(benign):
        raise ValueError("false-positive budget must be in [0, benign rows)")
    if not np.all(np.isfinite(benign)) or not np.all(np.isfinite(malicious)):
        raise ValueError("threshold calibration scores must be finite")
    weakest = float(np.min(malicious))
    unavoidable = int(np.sum(benign >= weakest))
    benign_below = benign[benign < weakest]
    highest_below = float(np.max(benign_below)) if len(benign_below) else None
    threshold = (
        _logit_midpoint(highest_below, weakest)
        if highest_below is not None
        else float(np.nextafter(np.float64(weakest), np.float64(-np.inf)))
    )
    false_positives = int(np.sum(benign >= threshold))
    false_negatives = int(np.sum(malicious < threshold))
    return {
        "threshold": threshold,
        "false_positive_budget": int(false_positive_budget),
        "weakest_malicious_score": weakest,
        "highest_benign_below_weakest_malicious": highest_below,
        "unavoidable_false_positives": unavoidable,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "feasible": bool(
            unavoidable <= false_positive_budget
            and false_positives <= false_positive_budget
            and false_negatives == 0
        ),
    }


@dataclass(frozen=True)
class ChannelCandidate:
    candidate_id: int
    channel: str
    classifier: Any
    validation_scores: np.ndarray
    learning_rate: float
    l2: float
    positive_weight: float


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    if positives == 0:
        raise ValueError("average precision requires a positive row")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(labels) + 1)
    return float(np.sum((cumulative / ranks) * sorted_labels) / positives)


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    statistic = float(np.sum(ranks[labels == 1])) - positives * (positives + 1) / 2
    return statistic / (positives * negatives)


def select_validation_pair(
    *,
    image_candidates: Sequence[ChannelCandidate],
    text_candidates: Sequence[ChannelCandidate],
    validation_labels: np.ndarray,
    validation_image_led: np.ndarray,
    validation_text_led: np.ndarray,
    budget_allocations: Iterable[tuple[int, int]] = ((0, 2), (1, 1), (2, 0)),
    combined_false_positive_limit: int = 2,
) -> tuple[dict[str, Any], ChannelCandidate, ChannelCandidate, list[dict[str, Any]]]:
    """Select a channel pair using only validation scores and OR behavior."""

    labels = np.asarray(validation_labels, dtype=np.int64)
    image_led = np.asarray(validation_image_led, dtype=bool)
    text_led = np.asarray(validation_text_led, dtype=bool)
    if labels.ndim != 1 or image_led.shape != labels.shape or text_led.shape != labels.shape:
        raise ValueError("validation labels/subgroup masks are not aligned")
    benign = labels == 0
    if not np.any(benign) or not np.any(image_led) or not np.any(text_led):
        raise ValueError("validation requires benign, image-led, and text-led rows")
    if np.any(image_led & text_led) or np.any(labels[image_led | text_led] != 1):
        raise ValueError("malicious channel masks are invalid")
    allocations = tuple((int(left), int(right)) for left, right in budget_allocations)
    if not allocations:
        raise ValueError("at least one threshold-budget allocation is required")
    candidates: list[dict[str, Any]] = []
    fitted_by_pair_id: dict[int, tuple[ChannelCandidate, ChannelCandidate]] = {}
    pair_id = 0
    for image_candidate in image_candidates:
        image_scores = np.asarray(image_candidate.validation_scores, dtype=np.float64)
        if image_scores.shape != labels.shape:
            raise ValueError("image candidate scores are not aligned")
        for text_candidate in text_candidates:
            text_scores = np.asarray(text_candidate.validation_scores, dtype=np.float64)
            if text_scores.shape != labels.shape:
                raise ValueError("text candidate scores are not aligned")
            for image_budget, text_budget in allocations:
                image_calibration = calibrate_channel_threshold(
                    image_scores[benign], image_scores[image_led], image_budget
                )
                text_calibration = calibrate_channel_threshold(
                    text_scores[benign], text_scores[text_led], text_budget
                )
                image_threshold = float(image_calibration["threshold"])
                text_threshold = float(text_calibration["threshold"])
                image_blocked = image_scores >= image_threshold
                text_blocked = text_scores >= text_threshold
                combined = image_blocked | text_blocked
                false_positives = int(np.sum(combined & benign))
                false_negatives = int(np.sum(~combined & (labels == 1)))
                image_false_negatives = int(np.sum(~image_blocked & image_led))
                text_false_negatives = int(np.sum(~text_blocked & text_led))
                risk = np.maximum(
                    logit(image_scores) - float(logit(image_threshold)),
                    logit(text_scores) - float(logit(text_threshold)),
                )
                precision_denominator = int(np.sum(combined))
                true_positives = int(np.sum(combined & (labels == 1)))
                precision = (
                    true_positives / precision_denominator if precision_denominator else 0.0
                )
                recall_denominator = int(np.sum(labels == 1))
                recall = true_positives / recall_denominator
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )
                worst_margin = float(
                    min(
                        np.min(logit(image_scores[image_led]) - logit(image_threshold)),
                        np.min(logit(text_scores[text_led]) - logit(text_threshold)),
                    )
                )
                feasible = bool(
                    image_calibration["feasible"]
                    and text_calibration["feasible"]
                    and image_false_negatives == 0
                    and text_false_negatives == 0
                    and false_negatives == 0
                    and false_positives <= combined_false_positive_limit
                )
                record = {
                    "pair_id": pair_id,
                    "image_candidate_id": image_candidate.candidate_id,
                    "text_candidate_id": text_candidate.candidate_id,
                    "image_learning_rate": image_candidate.learning_rate,
                    "image_l2": image_candidate.l2,
                    "image_positive_weight": image_candidate.positive_weight,
                    "text_learning_rate": text_candidate.learning_rate,
                    "text_l2": text_candidate.l2,
                    "text_positive_weight": text_candidate.positive_weight,
                    "image_false_positive_budget": image_budget,
                    "text_false_positive_budget": text_budget,
                    "image_threshold": image_threshold,
                    "text_threshold": text_threshold,
                    "image_unavoidable_false_positives": image_calibration[
                        "unavoidable_false_positives"
                    ],
                    "text_unavoidable_false_positives": text_calibration[
                        "unavoidable_false_positives"
                    ],
                    "validation_image_head_benign_false_positives": int(
                        np.sum(image_blocked & benign)
                    ),
                    "validation_text_head_benign_false_positives": int(
                        np.sum(text_blocked & benign)
                    ),
                    "validation_benign_false_positive_overlap": int(
                        np.sum(image_blocked & text_blocked & benign)
                    ),
                    "validation_false_positives": false_positives,
                    "validation_false_negatives": false_negatives,
                    "validation_image_led_false_negatives": image_false_negatives,
                    "validation_text_led_false_negatives": text_false_negatives,
                    "validation_false_positive_rate": false_positives / int(np.sum(benign)),
                    "validation_f1": f1,
                    "validation_auprc": _average_precision(labels, risk),
                    "validation_auroc": _auroc(labels, risk),
                    "validation_rank_separation": float(
                        np.min(risk[labels == 1]) - np.max(risk[benign])
                    ),
                    "validation_worst_relevant_channel_logit_margin": worst_margin,
                    "validation_feasible": feasible,
                    "selected": False,
                }
                candidates.append(record)
                fitted_by_pair_id[pair_id] = (image_candidate, text_candidate)
                pair_id += 1
    feasible_candidates = [row for row in candidates if row["validation_feasible"]]
    if not feasible_candidates:
        raise RuntimeError("no dual-head candidate pair meets the validation OR gates")
    selected = min(
        feasible_candidates,
        key=lambda row: (
            int(row["validation_false_positives"]),
            -float(row["validation_worst_relevant_channel_logit_margin"]),
            -float(row["validation_auprc"]),
            -float(row["validation_auroc"]),
            -float(row["validation_rank_separation"]),
            float(row["image_positive_weight"]) + float(row["text_positive_weight"]),
            -float(row["image_l2"]) - float(row["text_l2"]),
            float(row["image_learning_rate"]),
            float(row["text_learning_rate"]),
            int(row["pair_id"]),
        ),
    )
    selected["selected"] = True
    image, text = fitted_by_pair_id[int(selected["pair_id"])]
    return selected, image, text, candidates


def pair_metrics(
    *,
    labels: np.ndarray,
    image_scores: np.ndarray,
    text_scores: np.ndarray,
    image_threshold: float,
    text_threshold: float,
    image_led: np.ndarray | None = None,
    text_led: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    image = np.asarray(image_scores, dtype=np.float64)
    text = np.asarray(text_scores, dtype=np.float64)
    if y.ndim != 1 or image.shape != y.shape or text.shape != y.shape:
        raise ValueError("pair score arrays are not aligned")
    image_blocked = image >= image_threshold
    text_blocked = text >= text_threshold
    blocked = image_blocked | text_blocked
    benign = y == 0
    malicious = y == 1
    tp = int(np.sum(blocked & malicious))
    tn = int(np.sum(~blocked & benign))
    fp = int(np.sum(blocked & benign))
    fn = int(np.sum(~blocked & malicious))
    payload: dict[str, Any] = {
        "rows": len(y),
        "malicious_rows": int(np.sum(malicious)),
        "benign_rows": int(np.sum(benign)),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "image_head_benign_false_positives": int(np.sum(image_blocked & benign)),
        "text_head_benign_false_positives": int(np.sum(text_blocked & benign)),
        "benign_false_positive_overlap": int(
            np.sum(image_blocked & text_blocked & benign)
        ),
    }
    if image_led is not None:
        mask = np.asarray(image_led, dtype=bool)
        if mask.shape != y.shape:
            raise ValueError("image-led mask is not aligned")
        payload["image_led_rows"] = int(np.sum(mask))
        payload["image_head_image_led_false_negatives"] = int(
            np.sum(~image_blocked & mask)
        )
        payload["combined_image_led_false_negatives"] = int(np.sum(~blocked & mask))
    if text_led is not None:
        mask = np.asarray(text_led, dtype=bool)
        if mask.shape != y.shape:
            raise ValueError("text-led mask is not aligned")
        payload["text_led_rows"] = int(np.sum(mask))
        payload["text_head_text_led_false_negatives"] = int(np.sum(~text_blocked & mask))
        payload["combined_text_led_false_negatives"] = int(np.sum(~blocked & mask))
    return payload
