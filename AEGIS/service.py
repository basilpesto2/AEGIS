from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from AEGIS.detector_artifact import DetectorArtifact
from AEGIS.guardrail import (
    EmbeddingProvider,
    GuardrailPolicy,
    GuardrailRequest,
    GuardrailRuntime,
    decision_summary,
)


@dataclass(frozen=True)
class RequestLimits:
    max_batch_size: int = 8
    max_text_characters: int = 32_768
    max_request_id_characters: int = 256
    max_images_per_request: int = 1
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    allowed_image_media_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
    )
    allow_local_image_paths: bool = False
    allowed_image_root: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_batch_size",
            "max_text_characters",
            "max_request_id_characters",
            "max_images_per_request",
            "max_image_bytes",
            "max_image_pixels",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.allowed_image_root is not None and not self.allow_local_image_paths:
            raise ValueError(
                "allowed_image_root requires allow_local_image_paths to be enabled."
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
    limits: RequestLimits | None = None,
    include_error_details: bool = False,
) -> dict[str, object]:
    """Evaluate JSON-style prompt request payloads through an embedding provider.

    Expected payload fields for a single request:
    - optional `text`
    - optional `image_path` or `image_paths`
    - optional `request_id`
    - optional `metadata` object

    Batch payloads can use `requests: [{...}, {...}]`.
    """

    effective_limits = limits or RequestLimits(allow_local_image_paths=True)
    request_payloads = payload.get("requests")
    if request_payloads is None:
        raw_requests = [payload]
    else:
        if not isinstance(request_payloads, list) or not request_payloads:
            raise ValueError("'requests' must be a non-empty list when provided.")
        raw_requests = request_payloads
    if len(raw_requests) > effective_limits.max_batch_size:
        raise ValueError(
            f"Batch contains {len(raw_requests)} requests; maximum is "
            f"{effective_limits.max_batch_size}."
        )

    with tempfile.TemporaryDirectory(prefix="aegis-service-request-") as directory:
        scratch = Path(directory)
        requests = [
            _request_from_payload(
                item,
                limits=effective_limits,
                scratch=scratch / f"request-{index:04d}",
            )
            for index, item in enumerate(raw_requests)
        ]
        runtime = GuardrailRuntime(
            artifact,
            policy=policy,
            include_error_details=include_error_details,
        )
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


def _request_from_payload(
    payload: dict[str, Any],
    *,
    limits: RequestLimits,
    scratch: Path,
) -> GuardrailRequest:
    if not isinstance(payload, dict):
        raise ValueError("Each request payload must be an object.")
    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("'metadata' must be an object when provided.")

    text = "" if payload.get("text") is None else str(payload.get("text"))
    if len(text) > limits.max_text_characters:
        raise ValueError(
            f"Request text contains {len(text)} characters; maximum is "
            f"{limits.max_text_characters}."
        )
    request_id = (
        None if payload.get("request_id") is None else str(payload.get("request_id"))
    )
    if request_id is not None and len(request_id) > limits.max_request_id_characters:
        raise ValueError(
            f"request_id contains {len(request_id)} characters; maximum is "
            f"{limits.max_request_id_characters}."
        )
    image_paths = _coerce_image_paths(payload, limits=limits, scratch=scratch)
    if not text.strip() and not image_paths:
        raise ValueError("Each request must contain text, an image, or both.")
    return GuardrailRequest(
        text=text,
        image_paths=image_paths,
        request_id=request_id,
        metadata=metadata,
    )


def _coerce_image_paths(
    payload: dict[str, Any],
    *,
    limits: RequestLimits,
    scratch: Path,
) -> tuple[str, ...]:
    encoded_images = payload.get("images")
    has_encoded = encoded_images is not None or payload.get("image_base64") is not None
    value = payload.get("image_paths", payload.get("image_path", ()))
    has_paths = value not in (None, (), [], "")
    if has_encoded and has_paths:
        raise ValueError("Provide encoded images or local image paths, not both.")
    if has_encoded:
        records = (
            encoded_images
            if encoded_images is not None
            else [
                {
                    "media_type": payload.get("image_media_type", "image/png"),
                    "base64": payload.get("image_base64"),
                }
            ]
        )
        if not isinstance(records, list) or not records:
            raise ValueError("'images' must be a non-empty list when provided.")
        if len(records) > limits.max_images_per_request:
            raise ValueError(
                f"Request contains {len(records)} images; maximum is "
                f"{limits.max_images_per_request}."
            )
        scratch.mkdir(parents=True, exist_ok=True)
        return tuple(
            str(_decode_image_record(record, index, scratch, limits))
            for index, record in enumerate(records)
        )
    if value is None:
        return tuple()
    if isinstance(value, str):
        paths = (value,)
    elif isinstance(value, list | tuple):
        paths = tuple(str(item) for item in value)
    else:
        raise ValueError("'image_paths' must be a string or list of strings.")
    if not paths:
        return tuple()
    if not limits.allow_local_image_paths:
        raise ValueError(
            "Local image paths are disabled; submit validated base64 image bytes instead."
        )
    if len(paths) > limits.max_images_per_request:
        raise ValueError(
            f"Request contains {len(paths)} images; maximum is "
            f"{limits.max_images_per_request}."
        )
    return tuple(_validate_local_image_path(path, limits) for path in paths)


def _decode_image_record(
    record: object,
    index: int,
    scratch: Path,
    limits: RequestLimits,
) -> Path:
    if not isinstance(record, dict):
        raise ValueError("Each image must be an object with 'media_type' and 'base64'.")
    media_type = str(record.get("media_type", ""))
    if media_type not in limits.allowed_image_media_types:
        raise ValueError(f"Unsupported image media_type: {media_type!r}.")
    encoded = record.get("base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("Each image must include a non-empty base64 string.")
    maximum_encoded = ((limits.max_image_bytes + 2) // 3) * 4
    if len(encoded) > maximum_encoded + 4:
        raise ValueError(f"Encoded image exceeds {limits.max_image_bytes} decoded bytes.")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image base64 is invalid.") from exc
    if len(content) > limits.max_image_bytes:
        raise ValueError(f"Image exceeds {limits.max_image_bytes} decoded bytes.")
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
            actual_format = str(image.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Decoded image is not a supported, valid image.") from exc
    if width <= 0 or height <= 0 or width * height > limits.max_image_pixels:
        raise ValueError(
            f"Decoded image dimensions {width}x{height} exceed the pixel limit "
            f"of {limits.max_image_pixels}."
        )
    suffixes = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
    expected_formats = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
    }
    if actual_format != expected_formats[media_type]:
        raise ValueError(
            f"Image content format {actual_format!r} does not match {media_type!r}."
        )
    output = scratch / f"image-{index:04d}{suffixes[actual_format]}"
    output.write_bytes(content)
    return output


def _validate_local_image_path(value: str, limits: RequestLimits) -> str:
    path = Path(value).expanduser().resolve()
    if limits.allowed_image_root is not None:
        root = Path(limits.allowed_image_root).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Image path is outside the allowed root: {path}") from exc
    if not path.exists() or not path.is_file():
        raise ValueError(f"Image path does not exist or is not a file: {path}")
    if path.stat().st_size > limits.max_image_bytes:
        raise ValueError(f"Image exceeds {limits.max_image_bytes} bytes: {path}")
    return str(path)
