from __future__ import annotations

import json
import math
from typing import Any


_DUPLICATE_KEY_MESSAGE = "Duplicate JSON object key is not permitted."
_NONFINITE_NUMBER_MESSAGE = "Non-finite JSON number is not permitted."


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    """Decode standards-compliant JSON while rejecting ambiguous objects."""

    return json.loads(
        value,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_finite_float,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(_DUPLICATE_KEY_MESSAGE)
        result[key] = value
    return result


def _reject_nonfinite_constant(_value: str) -> None:
    raise ValueError(_NONFINITE_NUMBER_MESSAGE)


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(_NONFINITE_NUMBER_MESSAGE)
    return parsed


__all__ = ["strict_json_loads"]
