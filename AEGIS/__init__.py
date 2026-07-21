"""Malicious-prompt detection experiments and inference tools."""

from AEGIS.client import (
    GuardedRequest,
    GuardrailBlocked,
    GuardrailClient,
    GuardrailClientError,
    GuardrailReviewRequired,
)


__all__ = [
    "GuardedRequest",
    "GuardrailBlocked",
    "GuardrailClient",
    "GuardrailClientError",
    "GuardrailReviewRequired",
    "__version__",
]

__version__ = "0.3.0"
