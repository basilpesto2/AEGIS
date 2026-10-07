from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]

sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(PIPELINE_ROOT))

from aegis_research.bordair_dual import (  # noqa: E402
    ACTION_ORDER,
    CACHE_POOLING,
    HEAD_POOLINGS,
    load_detector_pair,
)
from aegis_research.signals import (  # noqa: E402
    POOLING_FEATURE_VIEWS,
    build_feature_views,
)


PROVENANCE_KEYS = (
    "model_family",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "preprocessing_sha256",
    "layer",
    "pooling",
)
SHARED_PROVENANCE_KEYS = PROVENANCE_KEYS[:-1]
DUAL_OUTPUT_FIELDS = (
    "detector_mode",
    "risk_score",
    "detector_threshold",
    "review_threshold",
    "verdict",
    "uncertain",
    "recommended_action",
    "decisive_head",
    "text_risk_score",
    "text_block_threshold",
    "text_review_threshold",
    "text_action",
    "text_pooling",
    "image_risk_score",
    "image_block_threshold",
    "image_review_threshold",
    "image_action",
    "image_pooling",
    "runtime_detector_identity_sha256",
    "pair_identity_sha256",
)
LEGACY_OUTPUT_FIELDS = (
    "detector_mode",
    "risk_score",
    "detector_threshold",
    "feature_view",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score a provenance-complete feature bundle with either a qualified "
            "AEGIS dual-head pair or an explicitly labelled legacy single head."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("dual_or", "legacy_single"),
        help=(
            "dual_or is the maintained v7 workflow; legacy_single must be "
            "selected explicitly for historical research or deployed artifacts."
        ),
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    detector = parser.add_mutually_exclusive_group(required=True)
    detector.add_argument(
        "--pair-manifest",
        type=Path,
        help="Qualified detector_pair_manifest.json used by dual_or mode.",
    )
    detector.add_argument(
        "--detector",
        type=Path,
        help="Historical research/deployed NPZ used by legacy_single mode.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing score table.",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.mode == "dual_or" and args.pair_manifest is None:
        parser.error("dual_or mode requires --pair-manifest")
    if args.mode == "legacy_single" and args.detector is None:
        parser.error("legacy_single mode requires --detector")
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"refusing to overwrite existing evidence: {args.output}; "
            "choose a new --output or use --force"
        )

    rows, id_column, views, feature_source, bundle_provenance = _load_bundle(
        args.metadata,
        args.features,
    )
    _reject_reserved_output_columns(
        rows[0],
        DUAL_OUTPUT_FIELDS if args.mode == "dual_or" else LEGACY_OUTPUT_FIELDS,
    )
    if args.mode == "dual_or":
        assert args.pair_manifest is not None
        output_rows, summary = _score_dual_or(
            rows=rows,
            views=views,
            bundle_provenance=bundle_provenance,
            pair_manifest_path=args.pair_manifest,
        )
    else:
        assert args.detector is not None
        output_rows, summary = _score_legacy_single(
            rows=rows,
            views=views,
            feature_source=feature_source,
            bundle_provenance=bundle_provenance,
            detector_path=args.detector,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "id_column": id_column,
                **summary,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _load_bundle(
    metadata_path: Path,
    features_path: Path,
) -> tuple[
    list[dict[str, str]],
    str,
    dict[str, np.ndarray],
    str,
    dict[str, object],
]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("metadata is empty")
    id_column = _id_column(rows[0])
    expected_ids = np.asarray([row[id_column] for row in rows])
    if any(not value.strip() for value in expected_ids):
        raise ValueError(f"{id_column} values must be non-empty")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError(f"{id_column} values must be unique")

    with np.load(features_path, allow_pickle=False) as bundle:
        required = {
            "sample_ids",
            "text_embeddings",
            "image_embeddings",
            "attribution_features",
            "feature_source",
            "id_column",
            *PROVENANCE_KEYS,
        }
        missing = required - set(bundle.files)
        if missing:
            raise ValueError(
                "feature bundle is missing required arrays or provenance: "
                + ", ".join(sorted(missing))
            )
        sample_ids = np.asarray(bundle["sample_ids"]).astype(str)
        bundle_id_column = _scalar_string(bundle, "id_column")
        if bundle_id_column not in {"sample_id", "variant_id"}:
            raise ValueError(
                "feature bundle id_column must be sample_id or variant_id"
            )
        if bundle_id_column != id_column:
            raise ValueError(
                "feature bundle id_column does not match metadata: "
                f"{bundle_id_column!r} != {id_column!r}"
            )
        if not np.array_equal(sample_ids, expected_ids):
            raise ValueError("feature IDs are not aligned to metadata")
        text_embeddings = np.asarray(
            bundle["text_embeddings"], dtype=np.float64
        )
        image_embeddings = np.asarray(
            bundle["image_embeddings"], dtype=np.float64
        )
        attribution_features = np.asarray(
            bundle["attribution_features"], dtype=np.float64
        )
        arrays = (text_embeddings, image_embeddings, attribution_features)
        if any(
            value.ndim != 2
            or len(value) != len(rows)
            or not np.all(np.isfinite(value))
            for value in arrays
        ):
            raise ValueError(
                "feature arrays must be finite 2D arrays aligned to metadata"
            )
        if text_embeddings.shape != image_embeddings.shape:
            raise ValueError(
                "text and image embedding matrices must have matching shapes"
            )
        views = build_feature_views(
            text_embeddings,
            image_embeddings,
            attribution_features,
        )
        feature_source = _scalar_string(bundle, "feature_source")
        bundle_provenance = _provenance(bundle)
    return rows, id_column, views, feature_source, bundle_provenance


def _score_dual_or(
    *,
    rows: list[dict[str, str]],
    views: Mapping[str, np.ndarray],
    bundle_provenance: Mapping[str, object],
    pair_manifest_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    text_features = np.asarray(views["text_representation"], dtype=np.float64)
    image_features = np.asarray(views["image_representation"], dtype=np.float64)
    if text_features.shape != image_features.shape:
        raise ValueError("dual primitive feature views must have matching shapes")
    pair = load_detector_pair(
        pair_manifest_path,
        expected_base_feature_dim=int(text_features.shape[1]),
    )
    qualification = pair.manifest.get("qualification_evidence")
    if not isinstance(qualification, Mapping) or qualification.get("passed") is not True:
        raise ValueError(
            "dual_or scoring requires a detector pair with passing, bound "
            "qualification evidence"
        )
    _validate_dual_bundle(
        bundle_provenance=bundle_provenance,
        text_features=text_features,
        image_features=image_features,
        manifest=pair.manifest,
    )

    artifacts = {
        "text": pair.text_artifact,
        "image": pair.image_artifact,
    }
    primitive_views = {
        "text": text_features,
        "image": image_features,
    }
    head_scores: dict[str, np.ndarray] = {}
    head_settings: dict[str, dict[str, object]] = {}
    for name in ("text", "image"):
        artifact = artifacts[name]
        entry = pair.manifest["artifacts"][name]
        block = float(artifact.threshold)
        review = float(entry["review_threshold"])
        if artifact.pooling != HEAD_POOLINGS[name]:
            raise ValueError(f"{name} artifact pooling does not select its primitive view")
        if int(artifact.feature_dim) != primitive_views[name].shape[1]:
            raise ValueError(
                f"{name} detector expects {artifact.feature_dim} primitive features, "
                f"but the bundle has {primitive_views[name].shape[1]}"
            )
        if block != float(entry["block_threshold"]):
            raise ValueError(f"{name} artifact and pair block thresholds differ")
        _validate_thresholds(block, review, name)
        scores = np.asarray(artifact.score(primitive_views[name]), dtype=np.float64)
        if scores.shape != (len(rows),) or not np.all(np.isfinite(scores)):
            raise ValueError(f"{name} detector returned invalid scores")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(f"{name} detector scores must be probabilities")
        head_scores[name] = scores
        head_settings[name] = {
            "block_threshold": block,
            "review_threshold": review,
            "pooling": artifact.pooling,
        }

    runtime_identity = str(pair.manifest["runtime_detector_identity_sha256"])
    pair_identity = str(pair.manifest["pair_identity_sha256"])
    output_rows: list[dict[str, object]] = []
    action_counts = {action: 0 for action in ACTION_ORDER}
    decisive_counts = {name: 0 for name in HEAD_POOLINGS}
    for index, row in enumerate(rows):
        decision = _compose_dual_decision(
            {
                name: {
                    "risk_score": float(head_scores[name][index]),
                    **head_settings[name],
                }
                for name in ("text", "image")
            }
        )
        action_counts[str(decision["recommended_action"])] += 1
        decisive_counts[str(decision["decisive_head"])] += 1
        output_rows.append(
            {
                **row,
                "detector_mode": "dual_or",
                "risk_score": decision["risk_score"],
                "detector_threshold": decision["detector_threshold"],
                "review_threshold": decision["review_threshold"],
                "verdict": decision["verdict"],
                "uncertain": decision["uncertain"],
                "recommended_action": decision["recommended_action"],
                "decisive_head": decision["decisive_head"],
                "text_risk_score": decision["heads"]["text"]["risk_score"],
                "text_block_threshold": decision["heads"]["text"][
                    "block_threshold"
                ],
                "text_review_threshold": decision["heads"]["text"][
                    "review_threshold"
                ],
                "text_action": decision["heads"]["text"]["action"],
                "text_pooling": decision["heads"]["text"]["pooling"],
                "image_risk_score": decision["heads"]["image"]["risk_score"],
                "image_block_threshold": decision["heads"]["image"][
                    "block_threshold"
                ],
                "image_review_threshold": decision["heads"]["image"][
                    "review_threshold"
                ],
                "image_action": decision["heads"]["image"]["action"],
                "image_pooling": decision["heads"]["image"]["pooling"],
                "runtime_detector_identity_sha256": runtime_identity,
                "pair_identity_sha256": pair_identity,
            }
        )
    if tuple(output_rows[0].keys())[-len(DUAL_OUTPUT_FIELDS) :] != DUAL_OUTPUT_FIELDS:
        raise RuntimeError("dual score output field contract changed")
    return output_rows, {
        "mode": "dual_or",
        "pair_manifest": str(pair.manifest_path),
        "pair_identity_sha256": pair_identity,
        "runtime_detector_identity_sha256": runtime_identity,
        "primitive_feature_dim": int(text_features.shape[1]),
        "fused_feature_dim": int(text_features.shape[1] + image_features.shape[1]),
        "bundle_pooling": bundle_provenance["pooling"],
        "heads": {
            name: {
                "feature_view": f"{name}_representation",
                **head_settings[name],
            }
            for name in ("text", "image")
        },
        "action_counts": action_counts,
        "decisive_head_counts": decisive_counts,
    }


def _compose_dual_decision(
    heads: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if set(heads) != set(HEAD_POOLINGS):
        raise ValueError("dual OR composition requires exactly text and image heads")
    normalized: dict[str, dict[str, object]] = {}
    ranked: list[tuple[int, float, str, dict[str, object]]] = []
    for name in ("text", "image"):
        entry = heads[name]
        score = float(entry["risk_score"])
        block = float(entry["block_threshold"])
        review = float(entry["review_threshold"])
        _validate_thresholds(block, review, name)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{name} risk score must be finite and in [0, 1]")
        if score >= block:
            action = "block"
            progress = (score - block) / (1.0 - block)
        elif score >= review:
            action = "review"
            progress = (score - review) / (block - review)
        else:
            action = "allow"
            progress = score / review if review > 0.0 else 0.0
        head = {
            "risk_score": score,
            "block_threshold": block,
            "review_threshold": review,
            "action": action,
            "pooling": str(entry["pooling"]),
        }
        normalized[name] = head
        ranked.append((ACTION_ORDER[action], progress, name, head))
    winning_rank = max(item[0] for item in ranked)
    candidates = [item for item in ranked if item[0] == winning_rank]
    _, _, decisive_name, decisive = sorted(
        candidates,
        key=lambda item: (-item[1], item[2]),
    )[0]
    action = str(decisive["action"])
    return {
        "risk_score": decisive["risk_score"],
        "detector_threshold": decisive["block_threshold"],
        "review_threshold": decisive["review_threshold"],
        "verdict": "malicious" if action == "block" else "benign",
        "uncertain": action == "review",
        "recommended_action": action,
        "decisive_head": decisive_name,
        "heads": normalized,
    }


def _validate_dual_bundle(
    *,
    bundle_provenance: Mapping[str, object],
    text_features: np.ndarray,
    image_features: np.ndarray,
    manifest: Mapping[str, Any],
) -> None:
    if bundle_provenance.get("pooling") != CACHE_POOLING:
        raise ValueError(
            "dual_or scoring requires a text_image_tokens feature bundle"
        )
    shared = manifest["shared_provenance"]
    expected = {key: shared[key] for key in SHARED_PROVENANCE_KEYS}
    actual = {key: bundle_provenance[key] for key in SHARED_PROVENANCE_KEYS}
    _validate_provenance(actual, expected, keys=SHARED_PROVENANCE_KEYS)
    base_dim = int(shared["base_feature_dim"])
    expected_shape = (len(text_features), base_dim)
    if text_features.shape != expected_shape or image_features.shape != expected_shape:
        raise ValueError(
            "dual primitive feature dimensions do not match the pair: "
            f"text={text_features.shape}, image={image_features.shape}, "
            f"expected={expected_shape}"
        )
    cache = manifest["cache_representation"]
    if (
        cache.get("pooling") != CACHE_POOLING
        or cache.get("ordering") != ["text_tokens", "image_tokens"]
        or cache.get("feature_dim") != base_dim * 2
    ):
        raise ValueError("pair fused feature provenance/dimension is not canonical")


def _score_legacy_single(
    *,
    rows: list[dict[str, str]],
    views: Mapping[str, np.ndarray],
    feature_source: str,
    bundle_provenance: Mapping[str, object],
    detector_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    with np.load(detector_path, allow_pickle=False) as detector:
        required_detector = {"weights", "mean", "scale", "bias", "threshold"}
        missing_detector = required_detector - set(detector.files)
        if missing_detector:
            raise ValueError(
                "detector is missing required arrays: "
                + ", ".join(sorted(missing_detector))
            )
        metadata: dict[str, object]
        if "metadata_json" in detector:
            metadata = json.loads(_scalar_string(detector, "metadata_json"))
            view_name = str(metadata.get("feature_view", ""))
            if not view_name:
                raise ValueError("research detector metadata lacks feature_view")
            if metadata.get("feature_source") != feature_source:
                raise ValueError(
                    "research detector feature_source does not match the bundle"
                )
        else:
            metadata = {}
            pooling = _scalar_string(detector, "pooling")
            try:
                view_name = POOLING_FEATURE_VIEWS[pooling]
            except KeyError as exc:
                raise ValueError(
                    "deployed detector pooling cannot be mapped to a feature view: "
                    f"{pooling!r}"
                ) from exc
        if view_name not in views:
            raise ValueError(f"unknown detector feature view: {view_name}")
        detector_provenance = _detector_provenance(detector, metadata)
        _validate_provenance(bundle_provenance, detector_provenance)
        selected_features = np.asarray(views[view_name], dtype=np.float64)
        weights = np.asarray(detector["weights"], dtype=np.float64).reshape(-1)
        mean = np.asarray(detector["mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(detector["scale"], dtype=np.float64).reshape(-1)
        bias = float(np.asarray(detector["bias"]).reshape(-1)[0])
        threshold = float(np.asarray(detector["threshold"]).reshape(-1)[0])

    if not (
        weights.shape == mean.shape == scale.shape == (selected_features.shape[1],)
    ):
        raise ValueError(
            f"detector expects {weights.size} features, but {view_name} has "
            f"{selected_features.shape[1]}"
        )
    if not all(
        np.all(np.isfinite(value)) for value in (weights, mean, scale)
    ) or not all(math.isfinite(value) for value in (bias, threshold)):
        raise ValueError("detector parameters must be finite")
    if np.any(scale <= 0):
        raise ValueError("detector scale values must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("detector threshold must be between zero and one")
    logits = ((selected_features - mean) / scale) @ weights + bias
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    output_rows = [
        {
            **row,
            "detector_mode": "legacy_single",
            "risk_score": float(score),
            "detector_threshold": threshold,
            "feature_view": view_name,
        }
        for row, score in zip(rows, scores)
    ]
    return output_rows, {
        "mode": "legacy_single",
        "feature_view": view_name,
        "feature_dim": selected_features.shape[1],
        "threshold": threshold,
        "detector_provenance": detector_provenance,
    }


def _validate_thresholds(block: float, review: float, name: str) -> None:
    if not math.isfinite(block) or not 0.0 < block < 1.0:
        raise ValueError(f"{name} block threshold must be finite and in (0, 1)")
    if not math.isfinite(review) or not 0.0 <= review < block:
        raise ValueError(
            f"{name} review threshold must be finite, non-negative, and below block"
        )


def _reject_reserved_output_columns(
    metadata_row: Mapping[str, object],
    output_fields: tuple[str, ...],
) -> None:
    collisions = sorted(set(metadata_row).intersection(output_fields))
    if collisions:
        raise ValueError(
            "metadata columns collide with reserved score output fields: "
            + ", ".join(collisions)
        )


def _id_column(row: dict[str, str]) -> str:
    if "sample_id" in row:
        return "sample_id"
    if "variant_id" in row:
        return "variant_id"
    raise ValueError("metadata must contain sample_id or variant_id")


def _scalar_string(data: np.lib.npyio.NpzFile, name: str) -> str:
    value = str(np.asarray(data[name]).reshape(-1)[0]).strip()
    if not value:
        raise ValueError(f"{name} provenance must be non-empty")
    return value


def _provenance(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    result: dict[str, object] = {
        key: (
            int(np.asarray(data[key]).reshape(-1)[0])
            if key == "layer"
            else _scalar_string(data, key)
        )
        for key in PROVENANCE_KEYS
    }
    fingerprint = str(result["preprocessing_sha256"])
    if (
        len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("preprocessing_sha256 must be lowercase SHA-256 hex")
    return result


def _detector_provenance(
    detector: np.lib.npyio.NpzFile,
    metadata: dict[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in PROVENANCE_KEYS:
        if key in detector:
            result[key] = (
                int(np.asarray(detector[key]).reshape(-1)[0])
                if key == "layer"
                else _scalar_string(detector, key)
            )
        elif key in metadata:
            result[key] = int(metadata[key]) if key == "layer" else str(metadata[key])
        else:
            raise ValueError(f"detector lacks required provenance: {key}")
    return result


def _validate_provenance(
    features: Mapping[str, object],
    detector: Mapping[str, object],
    *,
    keys: tuple[str, ...] = PROVENANCE_KEYS,
) -> None:
    mismatches: list[str] = []
    for key in keys:
        feature_value = features[key]
        detector_value = detector[key]
        if key == "model_id":
            feature_value = str(feature_value).replace("\\", "/")
            detector_value = str(detector_value).replace("\\", "/")
        if feature_value != detector_value:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "feature provenance does not match detector: "
            + ", ".join(mismatches)
        )


if __name__ == "__main__":
    main()
