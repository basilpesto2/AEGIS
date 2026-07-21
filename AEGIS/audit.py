from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any
import uuid


AUDIT_SCHEMA_VERSION = 1
VALID_TRAFFIC_MODES = {"shadow", "review", "enforce"}


@dataclass(frozen=True)
class AuditLogConfig:
    path: Path | None = None
    max_bytes: int = 100 * 1024 * 1024
    backup_count: int = 5
    fsync: bool = False

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError("audit.max_bytes must be positive.")
        if self.backup_count < 0:
            raise ValueError("audit.backup_count must be non-negative.")


def validate_traffic_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in VALID_TRAFFIC_MODES:
        raise ValueError(
            f"traffic_mode must be one of {sorted(VALID_TRAFFIC_MODES)}."
        )
    return mode


def apply_traffic_mode(
    response: dict[str, object],
    traffic_mode: str,
) -> dict[str, object]:
    """Apply rollout enforcement while retaining the detector recommendation."""

    mode = validate_traffic_mode(traffic_mode)
    decisions = response.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("Guardrail response decisions must be a list.")
    effective_counts: Counter[str] = Counter()
    recommended_counts: Counter[str] = Counter()
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("Guardrail response decisions must be objects.")
        recommended = str(decision.get("recommended_action") or decision.get("action"))
        if recommended not in {"allow", "review", "block"}:
            raise ValueError(f"Invalid guardrail action: {recommended!r}.")
        if mode == "shadow":
            enforced = "allow"
        elif mode == "review" and recommended == "block":
            enforced = "review"
        else:
            enforced = recommended
        decision["recommended_action"] = recommended
        decision["enforcement_action"] = enforced
        decision["action"] = enforced
        decision["traffic_mode"] = mode
        recommended_counts[recommended] += 1
        effective_counts[enforced] += 1

    summary = response.get("summary")
    if not isinstance(summary, dict):
        summary = {}
        response["summary"] = summary
    summary["traffic_mode"] = mode
    summary["recommended_action_counts"] = dict(sorted(recommended_counts.items()))
    summary["action_counts"] = dict(sorted(effective_counts.items()))
    return response


class PrivacySafeAuditLogger:
    """Append bounded, rotating decision records without raw request content."""

    def __init__(
        self,
        config: AuditLogConfig,
        *,
        context: dict[str, str | None] | None = None,
    ) -> None:
        if config.path is None:
            raise ValueError("Audit log path is required.")
        self.config = config
        self.path = config.path
        self.context = {"evidence_session_id": None, **dict(context or {})}
        self._lock = Lock()

    def readiness_context(self) -> dict[str, str | None]:
        return {
            "evidence_session_id": self.context.get("evidence_session_id"),
            "audit_path": str(self.path),
            "deployment_config_sha256": self.context.get(
                "deployment_config_sha256"
            ),
            "detector_sha256": self.context.get("detector_sha256"),
        }

    def record_response(
        self,
        response: dict[str, object],
        *,
        traffic_mode: str,
        status: int,
        duration_seconds: float,
    ) -> list[str]:
        decisions = response.get("decisions", [])
        if not isinstance(decisions, list):
            return []
        timestamp = datetime.now(timezone.utc).isoformat()
        events = []
        event_ids = []
        audited_decisions: list[tuple[dict[str, Any], str]] = []
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            event_id = str(uuid.uuid4())
            event_ids.append(event_id)
            audited_decisions.append((decision, event_id))
            events.append(
                _audit_event(
                    event_id=event_id,
                    timestamp=timestamp,
                    decision=decision,
                    traffic_mode=traffic_mode,
                    status=status,
                    duration_seconds=duration_seconds,
                    context=self.context,
                )
            )
        if events:
            self._append(events)
            for decision, event_id in audited_decisions:
                decision["audit_event_id"] = event_id
        return event_ids

    def _append(self, events: list[dict[str, object]]) -> None:
        encoded = b"".join(
            json.dumps(event, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
            for event in events
        )
        if len(encoded) > self.config.max_bytes:
            raise ValueError(
                "One audit response exceeds audit.max_bytes; reduce the batch size "
                "or increase the bounded audit limit."
            )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            current_size = self.path.stat().st_size if self.path.exists() else 0
            if current_size and current_size + len(encoded) > self.config.max_bytes:
                self._rotate()
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            with os.fdopen(descriptor, "ab", closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                if self.config.fsync:
                    os.fsync(handle.fileno())
            try:
                self.path.chmod(0o600)
            except OSError:
                pass

    def _rotate(self) -> None:
        if self.config.backup_count == 0:
            self.path.unlink(missing_ok=True)
            return
        oldest = _backup_path(self.path, self.config.backup_count)
        oldest.unlink(missing_ok=True)
        for index in range(self.config.backup_count - 1, 0, -1):
            source = _backup_path(self.path, index)
            if source.exists():
                source.replace(_backup_path(self.path, index + 1))
        self.path.replace(_backup_path(self.path, 1))


def _audit_event(
    *,
    event_id: str,
    timestamp: str,
    decision: dict[str, Any],
    traffic_mode: str,
    status: int,
    duration_seconds: float,
    context: dict[str, str | None],
) -> dict[str, object]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "event_id": event_id,
        "timestamp_utc": timestamp,
        "traffic_mode": validate_traffic_mode(traffic_mode),
        "http_status": int(status),
        "latency_seconds": float(duration_seconds),
        "action": decision.get("action"),
        "recommended_action": decision.get("recommended_action", decision.get("action")),
        "enforcement_action": decision.get("enforcement_action", decision.get("action")),
        "verdict": decision.get("verdict"),
        "risk_score": decision.get("risk_score"),
        "threshold": decision.get("threshold"),
        "review_threshold": decision.get("review_threshold"),
        "uncertain": bool(decision.get("uncertain")),
        "reasons": [str(value) for value in decision.get("reasons", [])],
        "fingerprint_algorithm": decision.get("fingerprint_algorithm"),
        "prompt_hmac_sha256": decision.get("prompt_hmac_sha256"),
        "image_hmac_sha256": [
            str(value) for value in decision.get("image_hmac_sha256", [])
        ],
        "modality": decision.get("modality"),
        "model_family": decision.get("model_family"),
        "model_id": decision.get("model_id"),
        "pooling": decision.get("pooling"),
        "detector_source": decision.get("detector_source"),
        "error_type": decision.get("error_type"),
        "deployment_name": context.get("deployment_name"),
        "target_profile": context.get("target_profile"),
        "evidence_session_id": context.get("evidence_session_id"),
        "deployment_config_sha256": context.get("deployment_config_sha256"),
        "detector_sha256": context.get("detector_sha256"),
    }


def _backup_path(path: Path, index: int) -> Path:
    return path.with_name(f"{path.name}.{index}")
