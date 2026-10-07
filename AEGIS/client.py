from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import math
from typing import Any, TypeVar
from urllib.parse import urlsplit
import uuid

from AEGIS.audit import validate_traffic_mode
from AEGIS.http_transport import is_loopback_hostname, request_with_deadline
from AEGIS.service import validate_request_payload_schema
from AEGIS.strict_json import strict_json_loads


_ACTIONS = {"allow", "review", "block"}
_VERDICTS = {"benign", "malicious", "guardrail_error"}
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_T = TypeVar("_T")


class GuardrailClientError(RuntimeError):
    """Base class for privacy-safe, fail-closed client errors."""

    code = "client_error"

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"


class GuardrailRequestError(GuardrailClientError):
    code = "invalid_request"


class GuardrailTransportError(GuardrailClientError):
    code = "transport_error"


class GuardrailAuthenticationError(GuardrailClientError):
    code = "authentication_error"


class GuardrailServiceError(GuardrailClientError):
    code = "service_error"


class GuardrailProtocolError(GuardrailClientError):
    code = "protocol_error"


class GuardrailTrafficModeError(GuardrailProtocolError):
    code = "traffic_mode_mismatch"


class GuardrailEvaluationError(GuardrailClientError):
    code = "guardrail_evaluation_error"


class GuardrailDecisionError(GuardrailClientError):
    def __init__(self, message: str, result: GuardrailResult) -> None:
        super().__init__(message)
        self.result = result


class GuardrailReviewRequired(GuardrailDecisionError):
    code = "review_required"


class GuardrailBlocked(GuardrailDecisionError):
    code = "blocked"


@dataclass(frozen=True)
class GuardrailDecision:
    action: str
    recommended_action: str
    traffic_mode: str
    verdict: str
    risk_score: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "recommended_action": self.recommended_action,
            "traffic_mode": self.traffic_mode,
            "verdict": self.verdict,
            "risk_score": self.risk_score,
        }


@dataclass(frozen=True)
class GuardrailResult:
    traffic_mode: str
    decisions: tuple[GuardrailDecision, ...]
    trace_id: str

    @property
    def action(self) -> str:
        actions = {item.action for item in self.decisions}
        if "block" in actions:
            return "block"
        if "review" in actions:
            return "review"
        return "allow"

    @property
    def downstream_allowed(self) -> bool:
        return self.action == "allow"

    def to_dict(self) -> dict[str, object]:
        return {
            "traffic_mode": self.traffic_mode,
            "action": self.action,
            "downstream_allowed": self.downstream_allowed,
            "trace_id": self.trace_id,
            "decisions": [item.to_dict() for item in self.decisions],
        }


@dataclass(frozen=True)
class GuardedRequest:
    """The exact JSON body evaluated by AEGIS, for downstream consumption."""

    _encoded_json: bytes = field(repr=False)

    def to_json_bytes(self) -> bytes:
        return bytes(self._encoded_json)

    def to_dict(self) -> dict[str, Any]:
        payload = strict_json_loads(self._encoded_json)
        if not isinstance(payload, dict):  # Guaranteed by construction.
            raise GuardrailProtocolError("Guarded request snapshot is invalid.")
        return payload

    def __repr__(self) -> str:
        return "GuardedRequest(redacted=True)"


class GuardrailClient:
    """Authenticated `/v1/guard` client with fail-closed callback gating."""

    __slots__ = (
        "_api_token",
        "_allow_unsafe_remote_http_for_development",
        "_base_url",
        "_required_traffic_mode",
        "_timeout_seconds",
    )

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8766",
        *,
        api_token: str | None = None,
        timeout_seconds: float = 30.0,
        required_traffic_mode: str | None = None,
        allow_unsafe_remote_http_for_development: bool = False,
    ) -> None:
        if type(allow_unsafe_remote_http_for_development) is not bool:
            raise TypeError(
                "allow_unsafe_remote_http_for_development must be a boolean."
            )
        self._allow_unsafe_remote_http_for_development = (
            allow_unsafe_remote_http_for_development
        )
        self._base_url = _validate_base_url(
            base_url,
            allow_unsafe_remote_http_for_development=(
                allow_unsafe_remote_http_for_development
            ),
        )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite.")
        self._api_token = None if api_token is None else str(api_token)
        if self._api_token is not None and any(
            character in self._api_token for character in ("\r", "\n")
        ):
            raise ValueError("api_token must not contain line breaks.")
        self._timeout_seconds = float(timeout_seconds)
        self._required_traffic_mode = (
            None
            if required_traffic_mode is None
            else validate_traffic_mode(required_traffic_mode)
        )

    def __repr__(self) -> str:
        return (
            "GuardrailClient("
            f"base_url={self._base_url!r}, "
            f"api_token_configured={self._api_token is not None!r}, "
            f"timeout_seconds={self._timeout_seconds!r}, "
            f"required_traffic_mode={self._required_traffic_mode!r}, "
            "allow_unsafe_remote_http_for_development="
            f"{self._allow_unsafe_remote_http_for_development!r})"
        )

    def guard(self, request_payload: Mapping[str, Any]) -> GuardrailResult:
        """Return a validated, content-free view of the effective AEGIS decision."""

        payload, expected_decisions = _encode_request(request_payload)
        return self._guard_encoded(payload, expected_decisions=expected_decisions)

    def _guard_encoded(
        self,
        payload: bytes,
        *,
        expected_decisions: int,
    ) -> GuardrailResult:
        trace_id = str(uuid.uuid4())
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Request-ID": trace_id,
        }
        if self._api_token is not None:
            headers["Authorization"] = f"Bearer {self._api_token}"
        try:
            response = request_with_deadline(
                self._base_url,
                "/v1/guard",
                method="POST",
                headers=headers,
                data=payload,
                timeout_seconds=self._timeout_seconds,
                max_response_bytes=_MAX_RESPONSE_BYTES,
            )
            status = response.status
            content_type = response.headers.get_content_type()
            response_trace_id = response.headers.get("X-Request-ID")
            body = response.body
        except (OSError, TimeoutError, ValueError):
            raise GuardrailTransportError(
                "AEGIS could not be reached; downstream inference was not invoked."
            ) from None

        if status in {401, 403}:
            raise GuardrailAuthenticationError(
                "AEGIS authentication failed; downstream inference was not invoked.",
                status=status,
            )
        if status != 200:
            raise GuardrailServiceError(
                "AEGIS returned a non-success status; downstream inference was not invoked.",
                status=status,
            )
        if content_type != "application/json" or len(body) > _MAX_RESPONSE_BYTES:
            raise GuardrailProtocolError(
                "AEGIS returned an invalid response; downstream inference was not invoked."
            )
        try:
            response_payload = strict_json_loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise GuardrailProtocolError(
                "AEGIS returned an invalid response; downstream inference was not invoked."
            ) from None
        result = _validate_response(
            response_payload,
            expected_decisions=expected_decisions,
            expected_trace_id=trace_id,
            response_trace_id=response_trace_id,
            required_traffic_mode=self._required_traffic_mode,
        )
        if any(item.verdict == "guardrail_error" for item in result.decisions):
            raise GuardrailEvaluationError(
                "AEGIS reported an evaluation error; downstream inference was not invoked."
            )
        return result

    def guarded_call(
        self,
        request_payload: Mapping[str, Any],
        downstream: Callable[[GuardedRequest], _T],
        /,
    ) -> _T:
        """Pass the exact evaluated request to downstream only on effective allow."""

        if not callable(downstream):
            raise GuardrailRequestError(
                "Downstream inference callback must be callable; it was not invoked."
            )
        if self._required_traffic_mode is None:
            raise GuardrailTrafficModeError(
                "guarded_call requires an explicit required_traffic_mode; downstream "
                "inference was not invoked."
            )
        encoded, expected_decisions = _encode_request(request_payload)
        _validate_guarded_call_snapshot(encoded)
        result = self._guard_encoded(
            encoded,
            expected_decisions=expected_decisions,
        )
        if result.action == "block":
            raise GuardrailBlocked(
                "AEGIS blocked the request; downstream inference was not invoked.",
                result,
            )
        if result.action == "review":
            raise GuardrailReviewRequired(
                "AEGIS requires review; downstream inference was not invoked.",
                result,
            )
        return downstream(GuardedRequest(encoded))


def _validate_base_url(
    value: str,
    *,
    allow_unsafe_remote_http_for_development: bool,
) -> str:
    normalized = str(value).strip()
    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise ValueError("base_url is not a valid HTTP(S) origin.") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "base_url must be an http(s) origin without credentials, path, query, or fragment."
        )
    if (
        parsed.scheme == "http"
        and not is_loopback_hostname(hostname)
        and not allow_unsafe_remote_http_for_development
    ):
        raise ValueError(
            "Cleartext HTTP is allowed only for loopback AEGIS origins. Use HTTPS for "
            "remote services; the unsafe remote-HTTP development override must never "
            "be enabled in production."
        )
    return normalized.rstrip("/")


def _is_loopback_hostname(hostname: str) -> bool:
    """Backward-compatible private alias for callers that imported it."""

    return is_loopback_hostname(hostname)


def _encode_request(payload: Mapping[str, Any]) -> tuple[bytes, int]:
    if not isinstance(payload, Mapping):
        raise GuardrailRequestError(
            "AEGIS request must be a JSON object; downstream inference was not invoked."
        )
    try:
        encoded = json.dumps(
            dict(payload),
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except Exception:
        raise GuardrailRequestError(
            "AEGIS request is not JSON serializable; downstream inference was not invoked."
        ) from None
    try:
        validated_payload = strict_json_loads(encoded)
        expected_decisions = len(validate_request_payload_schema(validated_payload))
    except (UnicodeDecodeError, ValueError) as exc:
        raise GuardrailRequestError(
            f"AEGIS request schema is invalid: {exc} Downstream inference was not invoked."
        ) from None
    if len(encoded) > _MAX_REQUEST_BYTES:
        raise GuardrailRequestError(
            "AEGIS request exceeds the client size limit; downstream inference was not "
            "invoked."
        )
    return encoded, expected_decisions


def _validate_guarded_call_snapshot(encoded: bytes) -> None:
    """Reject fields that AEGIS does not score or cannot snapshot by value."""

    payload = strict_json_loads(encoded)
    request_payloads = payload.get("requests")
    requests = request_payloads if request_payloads is not None else [payload]
    for index, request in enumerate(requests, start=1):
        unevaluated_fields = sorted({"metadata", "request_id"} & set(request))
        if unevaluated_fields:
            raise GuardrailRequestError(
                f"AEGIS guarded_call request {index} must not contain unevaluated "
                f"fields {unevaluated_fields} because they are not detector input; "
                "downstream inference was not invoked."
            )
        path_fields = sorted({"image_path", "image_paths"} & set(request))
        if path_fields:
            raise GuardrailRequestError(
                f"AEGIS guarded_call request {index} must snapshot images as encoded "
                f"bytes, not mutable local-path fields {path_fields}; downstream "
                "inference was not invoked."
            )


def _validate_response(
    payload: object,
    *,
    expected_decisions: int,
    expected_trace_id: str,
    response_trace_id: str | None,
    required_traffic_mode: str | None,
) -> GuardrailResult:
    if not isinstance(payload, dict):
        raise _protocol_error()
    trace_id = payload.get("trace_id")
    if trace_id != expected_trace_id or response_trace_id != expected_trace_id:
        raise _protocol_error()
    raw_decisions = payload.get("decisions")
    summary = payload.get("summary")
    if (
        not isinstance(raw_decisions, list)
        or len(raw_decisions) != expected_decisions
        or not isinstance(summary, dict)
    ):
        raise _protocol_error()
    try:
        mode = validate_traffic_mode(summary.get("traffic_mode"))
    except (TypeError, ValueError):
        raise _protocol_error() from None
    if required_traffic_mode is not None and mode != required_traffic_mode:
        raise GuardrailTrafficModeError(
            "AEGIS traffic mode does not match the required mode; downstream inference "
            "was not invoked."
        )

    decisions: list[GuardrailDecision] = []
    action_counts: Counter[str] = Counter()
    recommended_counts: Counter[str] = Counter()
    for raw in raw_decisions:
        if not isinstance(raw, dict):
            raise _protocol_error()
        action = raw.get("action")
        enforcement = raw.get("enforcement_action")
        recommended = raw.get("recommended_action")
        decision_mode = raw.get("traffic_mode")
        verdict = raw.get("verdict")
        if (
            action not in _ACTIONS
            or enforcement != action
            or recommended not in _ACTIONS
            or decision_mode != mode
            or verdict not in _VERDICTS
            or action != _effective_action(str(recommended), mode)
        ):
            raise _protocol_error()
        risk_score = raw.get("risk_score")
        if verdict == "guardrail_error":
            if risk_score is not None:
                raise _protocol_error()
            parsed_risk = None
        else:
            if isinstance(risk_score, bool) or not isinstance(risk_score, (int, float)):
                raise _protocol_error()
            parsed_risk = float(risk_score)
            if not math.isfinite(parsed_risk) or not 0.0 <= parsed_risk <= 1.0:
                raise _protocol_error()
        decisions.append(
            GuardrailDecision(
                action=str(action),
                recommended_action=str(recommended),
                traffic_mode=mode,
                verdict=str(verdict),
                risk_score=parsed_risk,
            )
        )
        action_counts[str(action)] += 1
        recommended_counts[str(recommended)] += 1
    if not _counts_match(summary.get("action_counts"), action_counts) or not _counts_match(
        summary.get("recommended_action_counts"), recommended_counts
    ):
        raise _protocol_error()
    return GuardrailResult(
        traffic_mode=mode,
        decisions=tuple(decisions),
        trace_id=expected_trace_id,
    )


def _effective_action(recommended: str, mode: str) -> str:
    if mode == "shadow":
        return "allow"
    if mode == "review" and recommended == "block":
        return "review"
    return recommended


def _counts_match(value: object, expected: Counter[str]) -> bool:
    if not isinstance(value, dict):
        return False
    normalized: dict[str, int] = {}
    for key, count in value.items():
        if key not in _ACTIONS or isinstance(count, bool) or not isinstance(count, int):
            return False
        if count <= 0:
            return False
        normalized[str(key)] = count
    return normalized == dict(expected)


def _protocol_error() -> GuardrailProtocolError:
    return GuardrailProtocolError(
        "AEGIS returned an invalid response; downstream inference was not invoked."
    )


__all__ = [
    "GuardrailAuthenticationError",
    "GuardrailBlocked",
    "GuardrailClient",
    "GuardrailClientError",
    "GuardrailDecision",
    "GuardrailEvaluationError",
    "GuardrailProtocolError",
    "GuardrailRequestError",
    "GuardrailResult",
    "GuardrailReviewRequired",
    "GuardrailServiceError",
    "GuardrailTrafficModeError",
    "GuardrailTransportError",
    "GuardedRequest",
]
