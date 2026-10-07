from __future__ import annotations

from email.message import Message
import re


_CANONICAL_CONTENT_LENGTH = re.compile(r"(?:0|[1-9][0-9]*)\Z", re.ASCII)


class RequestFramingError(ValueError):
    """A sanitized rejection of ambiguous or unsupported HTTP framing."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def parse_request_content_length(
    headers: Message,
    *,
    body_required: bool,
) -> int:
    """Validate request framing and return the canonical content length.

    Body-required endpoints accept exactly one positive canonical Content-Length.
    Bodyless endpoints accept no Content-Length or exactly one literal ``0``.
    Transfer-Encoding is never accepted.
    """

    if type(body_required) is not bool:
        raise TypeError("body_required must be a boolean.")
    if headers.get_all("Transfer-Encoding", []):
        raise RequestFramingError(
            "transfer_encoding",
            "Transfer-Encoding is not permitted.",
        )

    values = headers.get_all("Content-Length", [])
    if len(values) > 1:
        raise RequestFramingError(
            "duplicate_content_length",
            "Multiple Content-Length fields are not permitted.",
        )
    if not values:
        if body_required:
            raise RequestFramingError(
                "missing_content_length",
                "Content-Length is required.",
            )
        return 0

    raw_value = values[0]
    if (
        not isinstance(raw_value, str)
        or _CANONICAL_CONTENT_LENGTH.fullmatch(raw_value) is None
    ):
        raise RequestFramingError(
            "invalid_content_length",
            "Content-Length must use canonical ASCII decimal syntax.",
        )
    try:
        length = int(raw_value, 10)
    except ValueError:
        raise RequestFramingError(
            "invalid_content_length",
            "Content-Length must use canonical ASCII decimal syntax.",
        ) from None

    if body_required and length == 0:
        raise RequestFramingError(
            "body_required",
            "A positive Content-Length is required.",
        )
    if not body_required and length != 0:
        raise RequestFramingError(
            "body_not_allowed",
            "A request body is not permitted.",
        )
    return length


__all__ = ["RequestFramingError", "parse_request_content_length"]
