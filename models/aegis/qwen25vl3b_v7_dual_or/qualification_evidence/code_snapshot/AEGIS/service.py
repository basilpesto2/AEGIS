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
from AEGIS.detector_set import OrDetector
from AEGIS.guardrail import (
    EmbeddingProvider,
    GuardrailPolicy,
    GuardrailRequest,
    GuardrailRuntime,
    decision_summary,
)

REQUEST_MODALITIES = ("text", "image", "image_text")
REQUEST_PAYLOAD_FIELDS = frozenset(
    {
        "text",
        "image_path",
        "image_paths",
        "request_id",
        "metadata",
        "images",
        "image_base64",
        "image_media_type",
    }
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
        if not isinstance(self.allow_local_image_paths, bool):
            raise ValueError("allow_local_image_paths must be a boolean.")
        for name in (
            "max_batch_size",
            "max_text_characters",
            "max_request_id_characters",
            "max_images_per_request",
            "max_image_bytes",
            "max_image_pixels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive.")
        supported_media_types = {"image/jpeg", "image/png", "image/webp"}
        if (
            not isinstance(self.allowed_image_media_types, (list, tuple))
            or not self.allowed_image_media_types
            or any(
                not isinstance(value, str) or value not in supported_media_types
                for value in self.allowed_image_media_types
            )
            or len(set(self.allowed_image_media_types))
            != len(self.allowed_image_media_types)
        ):
            raise ValueError(
                "allowed_image_media_types must be a non-empty unique sequence "
                "containing only image/jpeg, image/png, or image/webp."
            )
        object.__setattr__(
            self,
            "allowed_image_media_types",
            tuple(self.allowed_image_media_types),
        )
        if self.allowed_image_root is not None and (
            not isinstance(self.allowed_image_root, str)
            or not self.allowed_image_root.strip()
        ):
            raise ValueError(
                "allowed_image_root must be a non-empty string when provided."
            )
        if self.allowed_image_root is not None and not self.allow_local_image_paths:
            raise ValueError(
                "allowed_image_root requires allow_local_image_paths to be enabled."
            )


def evaluate_feature_payload(
    artifact: DetectorArtifact | OrDetector,
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
    artifact: DetectorArtifact | OrDetector,
    provider: EmbeddingProvider,
    payload: dict[str, Any],
    policy: GuardrailPolicy | None = None,
    limits: RequestLimits | None = None,
    include_error_details: bool = False,
    input_modalities: tuple[str, ...] | None = None,
    target_profile: str | None = None,
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
    effective_modalities = normalize_input_modalities(input_modalities)
    raw_requests = validate_request_payload_schema(payload)
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
        if effective_modalities is not None:
            for index, request in enumerate(requests):
                if request.modality not in effective_modalities:
                    profile_detail = (
                        f" for target profile {target_profile!r}"
                        if target_profile is not None
                        else ""
                    )
                    raise ValueError(
                        f"Request {index + 1} uses unsupported modality "
                        f"{request.modality!r}{profile_detail}; supported input "
                        f"modalities are {', '.join(effective_modalities)}."
                    )
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


def validate_request_payload_schema(payload: object) -> list[dict[str, Any]]:
    """Validate and return the unambiguous request objects in an API payload.

    A batch envelope contains only ``requests``. A single-request payload contains
    only fields that are actually consumed by :func:`evaluate_request_payload`.
    Rejecting unknown and conflicting aliases prevents callers from attaching
    unevaluated content that could later be forwarded to a downstream model.
    """

    if not isinstance(payload, dict):
        raise ValueError("JSON request body must be an object.")
    if "requests" in payload:
        unknown = sorted(set(payload) - {"requests"})
        if unknown:
            raise ValueError(
                "Batch request envelope contains unevaluated fields: "
                f"{unknown}. The envelope may contain only 'requests'."
            )
        requests = payload["requests"]
        if not isinstance(requests, list) or not requests:
            raise ValueError("'requests' must be a non-empty list when provided.")
        raw_requests = requests
    else:
        raw_requests = [payload]

    for index, request in enumerate(raw_requests, start=1):
        _validate_request_object_schema(request, index=index)
    return raw_requests


def _validate_request_object_schema(payload: object, *, index: int) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Request {index} must be an object.")
    unknown = sorted(set(payload) - REQUEST_PAYLOAD_FIELDS)
    if unknown:
        raise ValueError(f"Request {index} contains unknown fields: {unknown}.")

    for field_name in ("text", "request_id"):
        value = payload.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Request {index} field '{field_name}' must be a string.")
    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"Request {index} field 'metadata' must be an object.")

    image_sources = [
        field_name
        for field_name in ("images", "image_base64", "image_path", "image_paths")
        if field_name in payload
    ]
    if len(image_sources) > 1:
        raise ValueError(
            f"Request {index} must use at most one image source; found "
            f"{image_sources}."
        )
    image_path = payload.get("image_path")
    if image_path is not None and not isinstance(image_path, str):
        raise ValueError(f"Request {index} field 'image_path' must be a string.")
    image_paths = payload.get("image_paths")
    if image_paths is not None and (
        not isinstance(image_paths, list)
        or any(not isinstance(value, str) for value in image_paths)
    ):
        raise ValueError(
            f"Request {index} field 'image_paths' must be a list of strings."
        )

    if "image_media_type" in payload and "image_base64" not in payload:
        raise ValueError(
            f"Request {index} field 'image_media_type' requires 'image_base64'."
        )
    image_base64 = payload.get("image_base64")
    if image_base64 is not None and not isinstance(image_base64, str):
        raise ValueError(f"Request {index} field 'image_base64' must be a string.")
    image_media_type = payload.get("image_media_type")
    if image_media_type is not None and not isinstance(image_media_type, str):
        raise ValueError(f"Request {index} field 'image_media_type' must be a string.")

    images = payload.get("images")
    if images is not None:
        if not isinstance(images, list) or not images:
            raise ValueError(
                f"Request {index} field 'images' must be a non-empty list."
            )
        for image_index, image in enumerate(images, start=1):
            if not isinstance(image, dict):
                raise ValueError(
                    f"Request {index} image {image_index} must be an object."
                )
            unknown_image_fields = sorted(set(image) - {"media_type", "base64"})
            if unknown_image_fields:
                raise ValueError(
                    f"Request {index} image {image_index} contains unknown fields: "
                    f"{unknown_image_fields}."
                )
            if set(image) != {"media_type", "base64"}:
                raise ValueError(
                    f"Request {index} image {image_index} must contain 'media_type' "
                    "and 'base64'."
                )
            if not isinstance(image["media_type"], str) or not isinstance(
                image["base64"], str
            ):
                raise ValueError(
                    f"Request {index} image {image_index} fields must be strings."
                )


def normalize_input_modalities(
    input_modalities: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if input_modalities is None:
        return None
    if isinstance(input_modalities, str):
        raise ValueError("input_modalities must be a sequence of modality names.")
    normalized = tuple(dict.fromkeys(str(value).strip() for value in input_modalities))
    if not normalized or any(not value for value in normalized):
        raise ValueError("input_modalities must contain at least one non-empty modality.")
    unsupported = tuple(
        modality for modality in normalized if modality not in REQUEST_MODALITIES
    )
    if unsupported:
        raise ValueError(
            f"Unsupported input modalities {unsupported!r}; expected values from "
            f"{REQUEST_MODALITIES!r}."
        )
    return normalized


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
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
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
