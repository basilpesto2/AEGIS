from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PIPELINE_ROOT))

from aegis_research.signals import build_feature_views  # noqa: E402


PROVENANCE_KEYS = (
    "model_family",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "preprocessing_sha256",
    "layer",
    "pooling",
)
POOLING_FEATURE_VIEWS = {
    "text_tokens": "text_representation",
    "image_tokens": "image_representation",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score a provenance-complete feature bundle with a compatible "
            "research or deployed AEGIS detector."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing score table.",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"refusing to overwrite existing evidence: {args.output}; "
            "choose a new --output or use --force"
        )

    with args.metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("metadata is empty")
    id_column = _id_column(rows[0])
    expected_ids = np.asarray([row[id_column] for row in rows])
    if any(not value.strip() for value in expected_ids):
        raise ValueError(f"{id_column} values must be non-empty")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError(f"{id_column} values must be unique")

    with np.load(args.features, allow_pickle=False) as bundle:
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

    with np.load(args.detector, allow_pickle=False) as detector:
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

        selected_features = views[view_name]
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
            "risk_score": float(score),
            "detector_threshold": threshold,
            "feature_view": view_name,
        }
        for row, score in zip(rows, scores)
    ]
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
                "feature_view": view_name,
                "feature_dim": selected_features.shape[1],
                "threshold": threshold,
                "detector_provenance": detector_provenance,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
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
    features: dict[str, object],
    detector: dict[str, object],
) -> None:
    mismatches: list[str] = []
    for key in PROVENANCE_KEYS:
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
