from __future__ import annotations

import hmac
import re
from collections import Counter
from collections.abc import Mapping


MIN_SECRET_CHARACTERS = 32
MIN_SECRET_DISTINCT_CHARACTERS = 12
_PLACEHOLDER_PREFIXES = (
    "changeme",
    "default",
    "example",
    "password",
    "replacewith",
    "secret",
    "token",
)


def validate_secret(
    value: str | None,
    *,
    name: str,
    required: bool,
) -> tuple[str | None, tuple[str, ...]]:
    """Validate one deployment secret without ever returning it in an error."""

    if value is None or not value:
        if required:
            return None, (f"{name} is missing.",)
        return None, ()
    problems: list[str] = []
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        problems.append(f"{name} must contain only visible ASCII characters.")
    if len(value) < MIN_SECRET_CHARACTERS:
        problems.append(
            f"{name} must contain at least {MIN_SECRET_CHARACTERS} characters."
        )
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    if any(normalized.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES):
        problems.append(f"{name} must not use an example or common placeholder value.")
    counts = Counter(value)
    dominated = bool(counts) and max(counts.values()) * 2 > len(value)
    if (
        len(counts) < MIN_SECRET_DISTINCT_CHARACTERS
        or dominated
        or _has_identical_run(value, minimum_run=8)
        or _has_simple_sequence(value, minimum_run=8)
        or _is_repeated_pattern(value)
    ):
        problems.append(f"{name} must be an independently generated random value.")
    return value, tuple(problems)


def validate_distinct_secrets(values: Mapping[str, str | None]) -> tuple[str, ...]:
    """Reject reuse between configured secrets using constant-time comparisons."""

    configured = [(name, value) for name, value in values.items() if value]
    problems: list[str] = []
    for index, (left_name, left_value) in enumerate(configured):
        assert left_value is not None
        for right_name, right_value in configured[index + 1 :]:
            assert right_value is not None
            if hmac.compare_digest(
                left_value.encode("utf-8"),
                right_value.encode("utf-8"),
            ):
                problems.append(f"{left_name} and {right_name} must be different secrets.")
    return tuple(problems)


def secret_matches(presented: str, configured: str) -> bool:
    """Compare bearer secrets without allowing malformed text to raise."""

    try:
        presented_bytes = presented.encode("ascii")
        configured_bytes = configured.encode("ascii")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(presented_bytes, configured_bytes)


def _is_repeated_pattern(value: str) -> bool:
    for width in range(1, min(16, len(value) // 2) + 1):
        if len(value) % width == 0 and value == value[:width] * (len(value) // width):
            return True
    return False


def _has_identical_run(value: str, *, minimum_run: int) -> bool:
    run = 1
    for index in range(1, len(value)):
        if value[index] == value[index - 1]:
            run += 1
            if run >= minimum_run:
                return True
        else:
            run = 1
    return False


def _has_simple_sequence(value: str, *, minimum_run: int) -> bool:
    run = 1
    direction = 0
    for index in range(1, len(value)):
        previous = value[index - 1]
        current = value[index]
        delta = ord(current) - ord(previous)
        if (
            previous.isascii()
            and current.isascii()
            and previous.isalnum()
            and current.isalnum()
            and delta in {-1, 1}
        ):
            if delta == direction:
                run += 1
            else:
                direction = delta
                run = 2
            if run >= minimum_run:
                return True
        else:
            direction = 0
            run = 1
    return False


__all__ = [
    "MIN_SECRET_CHARACTERS",
    "MIN_SECRET_DISTINCT_CHARACTERS",
    "secret_matches",
    "validate_distinct_secrets",
    "validate_secret",
]
