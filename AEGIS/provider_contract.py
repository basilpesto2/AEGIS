from __future__ import annotations

import importlib
from typing import Any


_PROVIDER_ATTRS = ("model_family", "model_id", "pooling", "feature_dim")
FUSED_TEXT_IMAGE_POOLING = "text_image_tokens"
FUSED_COMPONENT_POOLINGS = ("text_tokens", "image_tokens")


def load_provider(
    spec: str,
    options: dict[str, object] | None = None,
) -> Any:
    """Load an embedding provider from a `module:object` reference."""

    if ":" not in spec:
        raise ValueError("Provider spec must use 'module:object'.")
    options = dict(options or {})
    module_name, object_name = spec.split(":", 1)
    value = getattr(importlib.import_module(module_name), object_name)
    if isinstance(value, type):
        return value(**options)
    if callable(value) and not all(hasattr(value, attr) for attr in _PROVIDER_ATTRS):
        return value(**options)
    if options:
        raise ValueError(
            "Provider options can only be used with a provider class or factory."
        )
    return value
