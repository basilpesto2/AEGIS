from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
from typing import Any
import uuid


_SCHEMA_VERSION = 1
_MAX_PAGE_SIZE = 100
_BUSY_TIMEOUT_MS = 10_000
_STATUSES = frozenset({"pending", "completed", "error"})
_ACTIONS = frozenset({"allow", "review", "block"})
_VERDICTS = frozenset({"benign", "malicious", "guardrail_error", "unknown"})
_MODALITIES = frozenset({"text", "image", "image_text"})


@dataclass(frozen=True)
class StoredImage:
    filename: str
    media_type: str
    content: bytes


class GUIHistoryStore:
    """Persistent, thread-safe request history for the optional AEGIS GUI.

    A new SQLite connection is opened for every public operation. This avoids sharing
    connection state between ``ThreadingHTTPServer`` request threads while WAL mode
    permits readers to continue during short writes.
    """

    def __init__(self, path: str | Path) -> None:
        if str(path) == ":memory:":
            raise ValueError("GUI history must use a persistent filesystem path.")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_pending(
        self,
        *,
        text: str,
        request_id: str | None,
        metadata: Mapping[str, Any] | None,
        images: Sequence[StoredImage] | None,
    ) -> str:
        """Persist an input and its images atomically before inference begins."""

        normalized_text = str(text)
        normalized_request_id = None if request_id is None else str(request_id)
        if metadata is None:
            normalized_metadata: dict[str, Any] = {}
        elif isinstance(metadata, Mapping):
            normalized_metadata = dict(metadata)
        else:
            raise TypeError("metadata must be an object or None.")
        normalized_images = [
            _normalize_image(image, position)
            for position, image in enumerate(images or ())
        ]

        record_id = str(uuid.uuid4())
        timestamp = _utc_now()
        modality = _input_modality(normalized_text, bool(normalized_images))
        input_payload = {
            "text": normalized_text,
            "request_id": normalized_request_id,
            "metadata": normalized_metadata,
            "images": [
                {
                    "position": position,
                    "filename": image.filename,
                    "media_type": image.media_type,
                    "size_bytes": len(image.content),
                }
                for position, image in enumerate(normalized_images)
            ],
        }
        input_json = _dump_json_object(input_payload, "input")

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO gui_history (
                    record_id,
                    created_at,
                    updated_at,
                    status,
                    text,
                    request_id,
                    input_json,
                    modality
                )
                VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)
                """,
                (
                    record_id,
                    timestamp,
                    timestamp,
                    normalized_text,
                    normalized_request_id,
                    input_json,
                    modality,
                ),
            )
            connection.executemany(
                """
                INSERT INTO gui_history_images (
                    record_id,
                    position,
                    filename,
                    media_type,
                    byte_size,
                    content
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record_id,
                        position,
                        image.filename,
                        image.media_type,
                        len(image.content),
                        sqlite3.Binary(image.content),
                    )
                    for position, image in enumerate(normalized_images)
                ],
            )
        return record_id

    def complete(
        self,
        record_id: str,
        response: dict[str, Any],
        duration_ms: float,
    ) -> None:
        """Finish a pending record with the full AEGIS response."""

        if not isinstance(response, dict):
            raise TypeError("response must be an object.")
        response_json = _dump_json_object(response, "response")
        duration = _normalize_duration(duration_ms)
        normalized = _normalized_response_fields(response)
        timestamp = _utc_now()

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE gui_history
                SET
                    updated_at = ?,
                    finished_at = ?,
                    status = 'completed',
                    response_json = ?,
                    error_json = NULL,
                    duration_ms = ?,
                    action = ?,
                    recommended_action = ?,
                    verdict = ?,
                    risk_score = ?,
                    traffic_mode = ?,
                    trace_id = ?,
                    model = ?,
                    model_family = ?,
                    model_id = ?,
                    modality = COALESCE(?, modality)
                WHERE record_id = ? AND status = 'pending'
                """,
                (
                    timestamp,
                    timestamp,
                    response_json,
                    duration,
                    normalized["action"],
                    normalized["recommended_action"],
                    normalized["verdict"],
                    normalized["risk_score"],
                    normalized["traffic_mode"],
                    normalized["trace_id"],
                    normalized["model"],
                    normalized["model_family"],
                    normalized["model_id"],
                    normalized["modality"],
                    str(record_id),
                ),
            )
            self._require_pending_transition(connection, cursor, str(record_id))

    def fail(
        self,
        record_id: str,
        error: dict[str, Any],
        duration_ms: float,
    ) -> None:
        """Finish a pending record with a structured transport or service error."""

        if not isinstance(error, dict):
            raise TypeError("error must be an object.")
        error_json = _dump_json_object(error, "error")
        duration = _normalize_duration(duration_ms)
        timestamp = _utc_now()
        trace_id = _error_trace_id(error)

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE gui_history
                SET
                    updated_at = ?,
                    finished_at = ?,
                    status = 'error',
                    response_json = NULL,
                    error_json = ?,
                    duration_ms = ?,
                    trace_id = ?
                WHERE record_id = ? AND status = 'pending'
                """,
                (
                    timestamp,
                    timestamp,
                    error_json,
                    duration,
                    trace_id,
                    str(record_id),
                ),
            )
            self._require_pending_transition(connection, cursor, str(record_id))

    def search(
        self,
        *,
        query: str = "",
        action: str | None = None,
        verdict: str | None = None,
        modality: str | None = None,
        status: str | None = None,
        date_from: str | date | datetime | None = None,
        date_to: str | date | datetime | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, object]:
        """Search history with whitelisted filters and bounded pagination."""

        page_size = _page_size(limit)
        page_offset = _page_offset(offset)
        normalized_action = _filter_value("action", action, _ACTIONS)
        normalized_verdict = _filter_value("verdict", verdict, _VERDICTS)
        normalized_modality = _filter_value("modality", modality, _MODALITIES)
        normalized_status = _filter_value("status", status, _STATUSES)

        conditions: list[str] = []
        parameters: list[object] = []
        search_text = str(query).strip().lower()
        if search_text:
            searchable_columns = (
                "h.text",
                "h.request_id",
                "h.input_json",
                "h.response_json",
                "h.error_json",
            )
            conditions.append(
                "("
                + " OR ".join(
                    f"instr(lower(COALESCE({column}, '')), ?) > 0"
                    for column in searchable_columns
                )
                + ")"
            )
            parameters.extend([search_text] * len(searchable_columns))

        for column, value in (
            ("action", normalized_action),
            ("verdict", normalized_verdict),
            ("modality", normalized_modality),
            ("status", normalized_status),
        ):
            if value is not None:
                conditions.append(f"h.{column} = ?")
                parameters.append(value)

        lower_bound = _date_bound(date_from, upper=False)
        upper_bound = _date_bound(date_to, upper=True)
        if lower_bound is not None:
            conditions.append("h.created_at >= ?")
            parameters.append(lower_bound.value)
        if upper_bound is not None:
            conditions.append(
                f"h.created_at {'<' if upper_bound.exclusive else '<='} ?"
            )
            parameters.append(upper_bound.value)
        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound.instant > upper_bound.instant
        ):
            raise ValueError("date_from must not be later than date_to.")

        where_sql = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._connection() as connection:
            connection.execute("BEGIN")
            total_row = connection.execute(
                "SELECT COUNT(*) AS total FROM gui_history AS h" + where_sql,
                parameters,
            ).fetchone()
            rows = connection.execute(
                """
                SELECT
                    h.record_id,
                    h.created_at,
                    h.updated_at,
                    h.finished_at,
                    h.status,
                    h.text,
                    h.request_id,
                    h.duration_ms,
                    h.action,
                    h.recommended_action,
                    h.verdict,
                    h.risk_score,
                    h.traffic_mode,
                    h.trace_id,
                    h.model,
                    h.model_family,
                    h.model_id,
                    h.modality,
                    h.error_json,
                    (
                        SELECT COUNT(*)
                        FROM gui_history_images AS image
                        WHERE image.record_id = h.record_id
                    ) AS image_count
                FROM gui_history AS h
                """
                + where_sql
                + " ORDER BY h.created_at DESC, h.record_id DESC LIMIT ? OFFSET ?",
                [*parameters, page_size, page_offset],
            ).fetchall()

        return {
            "items": [_list_item(row) for row in rows],
            "total": int(total_row["total"]) if total_row is not None else 0,
            "limit": page_size,
            "offset": page_offset,
        }

    def summary(self) -> dict[str, object]:
        """Return aggregate dashboard counts without exposing request content."""

        with self._connection() as connection:
            connection.execute("BEGIN")
            aggregate = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    AVG(risk_score) AS average_risk_score,
                    AVG(duration_ms) AS average_duration_ms,
                    MAX(created_at) AS latest_at
                FROM gui_history
                """
            ).fetchone()
            status_counts = _grouped_counts(connection, "status")
            action_counts = _grouped_counts(connection, "action")
            recommended_action_counts = _grouped_counts(
                connection,
                "recommended_action",
            )
            verdict_counts = _grouped_counts(connection, "verdict")
            modality_counts = _grouped_counts(connection, "modality")

        return {
            "total": int(aggregate["total"]) if aggregate is not None else 0,
            "status_counts": status_counts,
            "action_counts": action_counts,
            "recommended_action_counts": recommended_action_counts,
            "verdict_counts": verdict_counts,
            "modality_counts": modality_counts,
            "average_risk_score": _optional_float(
                None if aggregate is None else aggregate["average_risk_score"]
            ),
            "average_duration_ms": _optional_float(
                None if aggregate is None else aggregate["average_duration_ms"]
            ),
            "latest_at": None if aggregate is None else aggregate["latest_at"],
        }

    def get(self, record_id: str) -> dict[str, object] | None:
        """Return one parsed record and image descriptors, never image bytes."""

        normalized_id = str(record_id)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT
                    h.*,
                    (
                        SELECT COUNT(*)
                        FROM gui_history_images AS image
                        WHERE image.record_id = h.record_id
                    ) AS image_count
                FROM gui_history AS h
                WHERE h.record_id = ?
                """,
                (normalized_id,),
            ).fetchone()
            if row is None:
                return None
            image_rows = connection.execute(
                """
                SELECT position, filename, media_type, byte_size
                FROM gui_history_images
                WHERE record_id = ?
                ORDER BY position ASC
                """,
                (normalized_id,),
            ).fetchall()

        images = [
            {
                "record_id": normalized_id,
                "position": int(image["position"]),
                "filename": str(image["filename"]),
                "media_type": str(image["media_type"]),
                "size_bytes": int(image["byte_size"]),
                "url": (
                    f"/api/history/{normalized_id}/images/"
                    f"{int(image['position'])}"
                ),
            }
            for image in image_rows
        ]
        fallback_input = {
            "text": str(row["text"]),
            "request_id": row["request_id"],
            "metadata": {},
        }
        input_payload = _parse_json_object(row["input_json"], fallback_input)
        input_payload["images"] = images

        item = _list_item(row)
        item.update(
            {
                "input": input_payload,
                "output": _parse_optional_json_object(row["response_json"]),
                "error": _parse_optional_json_object(row["error_json"]),
                "images": images,
            }
        )
        return item

    def get_image(self, record_id: str, position: int) -> StoredImage | None:
        """Return one image body for an authenticated GUI image route."""

        if isinstance(position, bool):
            return None
        try:
            normalized_position = int(position)
        except (TypeError, ValueError):
            return None
        if normalized_position < 0:
            return None
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT filename, media_type, content
                FROM gui_history_images
                WHERE record_id = ? AND position = ?
                """,
                (str(record_id), normalized_position),
            ).fetchone()
        if row is None:
            return None
        return StoredImage(
            filename=str(row["filename"]),
            media_type=str(row["media_type"]),
            content=bytes(row["content"]),
        )

    def clear_all(self) -> dict[str, int]:
        """Delete every history record and its stored images atomically."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            record_row = connection.execute(
                "SELECT COUNT(*) AS total FROM gui_history"
            ).fetchone()
            image_row = connection.execute(
                "SELECT COUNT(*) AS total FROM gui_history_images"
            ).fetchone()
            connection.execute("DELETE FROM gui_history")

        return {
            "deleted_records": int(record_row["total"]) if record_row is not None else 0,
            "deleted_images": int(image_row["total"]) if image_row is not None else 0,
        }

    def _initialize(self) -> None:
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        try:
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA foreign_keys = ON")
            mode_row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            if mode_row is None or str(mode_row[0]).lower() != "wal":
                raise RuntimeError("SQLite could not enable WAL mode for GUI history.")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS gui_history (
                    record_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL
                        CHECK (status IN ('pending', 'completed', 'error')),
                    text TEXT NOT NULL,
                    request_id TEXT,
                    input_json TEXT NOT NULL,
                    response_json TEXT,
                    error_json TEXT,
                    duration_ms REAL
                        CHECK (duration_ms IS NULL OR duration_ms >= 0),
                    action TEXT,
                    recommended_action TEXT,
                    verdict TEXT,
                    risk_score REAL,
                    traffic_mode TEXT,
                    trace_id TEXT,
                    model TEXT,
                    model_family TEXT,
                    model_id TEXT,
                    modality TEXT NOT NULL
                        CHECK (modality IN ('text', 'image', 'image_text'))
                );

                CREATE TABLE IF NOT EXISTS gui_history_images (
                    record_id TEXT NOT NULL,
                    position INTEGER NOT NULL CHECK (position >= 0),
                    filename TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
                    content BLOB NOT NULL,
                    PRIMARY KEY (record_id, position),
                    FOREIGN KEY (record_id)
                        REFERENCES gui_history(record_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_gui_history_created_at
                    ON gui_history(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_gui_history_status_created
                    ON gui_history(status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_gui_history_action_created
                    ON gui_history(action, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_gui_history_verdict_created
                    ON gui_history(verdict, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_gui_history_modality_created
                    ON gui_history(modality, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_gui_history_request_id
                    ON gui_history(request_id);
                CREATE INDEX IF NOT EXISTS idx_gui_history_images_record
                    ON gui_history_images(record_id, position);
                """
            )
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.commit()
        finally:
            connection.close()
            self._harden_permissions()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._harden_permissions()

    def _harden_permissions(self) -> None:
        mode = stat.S_IRUSR | stat.S_IWUSR
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{self.path}{suffix}")
            try:
                if candidate.exists():
                    os.chmod(candidate, mode)
            except OSError:
                pass

    @staticmethod
    def _require_pending_transition(
        connection: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        record_id: str,
    ) -> None:
        if cursor.rowcount == 1:
            return
        row = connection.execute(
            "SELECT status FROM gui_history WHERE record_id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown GUI history record: {record_id}")
        raise ValueError(
            f"GUI history record {record_id} is already {row['status']}; "
            "only pending records can be finished."
        )


@dataclass(frozen=True)
class _DateBound:
    value: str
    instant: datetime
    exclusive: bool


def _normalize_image(image: StoredImage, position: int) -> StoredImage:
    if not isinstance(image, StoredImage):
        raise TypeError(f"images[{position}] must be a StoredImage.")
    raw_filename = str(image.filename).replace("\x00", "")
    filename = raw_filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not filename:
        filename = f"image-{position}"
    media_type = str(image.media_type).strip().lower()
    if not media_type or "\x00" in media_type:
        raise ValueError(f"images[{position}] has an invalid media type.")
    if not isinstance(image.content, (bytes, bytearray, memoryview)):
        raise TypeError(f"images[{position}].content must be bytes-like.")
    return StoredImage(
        filename=filename,
        media_type=media_type,
        content=bytes(image.content),
    )


def _input_modality(text: str, has_images: bool) -> str:
    if text.strip() and has_images:
        return "image_text"
    if has_images:
        return "image"
    return "text"


def _normalized_response_fields(response: Mapping[str, Any]) -> dict[str, object]:
    decisions = response.get("decisions")
    decision: Mapping[str, Any] = {}
    if isinstance(decisions, list) and decisions and isinstance(decisions[0], Mapping):
        decision = decisions[0]
    summary = response.get("summary")
    if not isinstance(summary, Mapping):
        summary = {}

    action = _enum_or_none(
        decision.get("action", decision.get("enforcement_action")),
        _ACTIONS,
    )
    recommended_action = _enum_or_none(
        decision.get("recommended_action", action),
        _ACTIONS,
    )
    verdict = _enum_or_none(decision.get("verdict"), _VERDICTS)
    modality = _enum_or_none(decision.get("modality"), _MODALITIES)
    traffic_mode = _optional_string(
        decision.get("traffic_mode", summary.get("traffic_mode"))
    )
    trace_id = _optional_string(response.get("trace_id"))
    model_family = _optional_string(decision.get("model_family"))
    model_id = _optional_string(decision.get("model_id"))
    model = _optional_string(decision.get("model")) or model_id or model_family
    return {
        "action": action,
        "recommended_action": recommended_action,
        "verdict": verdict,
        "risk_score": _risk_score(decision.get("risk_score")),
        "traffic_mode": traffic_mode,
        "trace_id": trace_id,
        "model": model,
        "model_family": model_family,
        "model_id": model_id,
        "modality": modality,
    }


def _error_trace_id(error: Mapping[str, Any]) -> str | None:
    direct = _optional_string(error.get("trace_id"))
    if direct is not None:
        return direct
    nested = error.get("error")
    if isinstance(nested, Mapping):
        return _optional_string(nested.get("trace_id"))
    return None


def _risk_score(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return None
    return parsed


def _normalize_duration(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("duration_ms must be a finite non-negative number.")
    try:
        duration = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("duration_ms must be a finite non-negative number.") from None
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration_ms must be a finite non-negative number.")
    return duration


def _page_size(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("limit must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("limit must be a positive integer.") from None
    if parsed <= 0:
        raise ValueError("limit must be a positive integer.")
    return min(parsed, _MAX_PAGE_SIZE)


def _page_offset(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("offset must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("offset must be a non-negative integer.") from None
    if parsed < 0:
        raise ValueError("offset must be a non-negative integer.")
    return parsed


def _filter_value(
    name: str,
    value: str | None,
    allowed: frozenset[str],
) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized not in allowed:
        raise ValueError(
            f"{name} must be one of {sorted(allowed)} when provided."
        )
    return normalized


def _date_bound(
    value: str | date | datetime | None,
    *,
    upper: bool,
) -> _DateBound | None:
    if value is None:
        return None
    date_only = False
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min, tzinfo=timezone.utc)
        date_only = True
    else:
        raw = str(value).strip()
        if not raw:
            return None
        if len(raw) == 10:
            try:
                parsed_date = date.fromisoformat(raw)
            except ValueError:
                pass
            else:
                parsed = datetime.combine(parsed_date, time.min, tzinfo=timezone.utc)
                date_only = True
        if not date_only:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(
                    "date filters must be ISO-8601 dates or timestamps."
                ) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    exclusive = bool(upper and date_only)
    if exclusive:
        parsed += timedelta(days=1)
    return _DateBound(
        value=_format_utc(parsed),
        instant=parsed,
        exclusive=exclusive,
    )


def _grouped_counts(
    connection: sqlite3.Connection,
    column: str,
) -> dict[str, int]:
    if column not in {
        "status",
        "action",
        "recommended_action",
        "verdict",
        "modality",
    }:
        raise ValueError("Unsupported grouped-count column.")
    rows = connection.execute(
        f"""
        SELECT {column} AS value, COUNT(*) AS count
        FROM gui_history
        WHERE {column} IS NOT NULL
        GROUP BY {column}
        ORDER BY {column}
        """
    ).fetchall()
    return {str(row["value"]): int(row["count"]) for row in rows}


def _list_item(row: sqlite3.Row) -> dict[str, object]:
    error = _parse_optional_json_object(row["error_json"])
    return {
        "id": str(row["record_id"]),
        "record_id": str(row["record_id"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "finished_at": row["finished_at"],
        "status": str(row["status"]),
        "request_id": row["request_id"],
        "text": str(row["text"]),
        "text_preview": _text_preview(str(row["text"])),
        "image_count": int(row["image_count"]),
        "duration_ms": _optional_float(row["duration_ms"]),
        "action": row["action"],
        "recommended_action": row["recommended_action"],
        "verdict": row["verdict"],
        "risk_score": _optional_float(row["risk_score"]),
        "traffic_mode": row["traffic_mode"],
        "trace_id": row["trace_id"],
        "model": row["model"],
        "model_family": row["model_family"],
        "model_id": row["model_id"],
        "modality": str(row["modality"]),
        "error_message": _error_message(error),
    }


def _error_message(error: Mapping[str, Any] | None) -> str | None:
    if error is None:
        return None
    nested = error.get("error")
    candidate = nested if isinstance(nested, Mapping) else error
    for key in ("message", "detail", "code"):
        value = _optional_string(candidate.get(key))
        if value is not None:
            return value
    return None


def _text_preview(value: str, maximum: int = 240) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= maximum:
        return normalized
    return normalized[: maximum - 1].rstrip() + "…"


def _dump_json_object(value: Mapping[str, Any], name: str) -> str:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be JSON serializable.") from None


def _parse_json_object(
    value: object,
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    parsed = _parse_optional_json_object(value)
    return dict(fallback) if parsed is None else parsed


def _parse_optional_json_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _enum_or_none(value: object, allowed: frozenset[str]) -> str | None:
    normalized = _optional_string(value)
    if normalized is None:
        return None
    normalized = normalized.lower()
    return normalized if normalized in allowed else None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _utc_now() -> str:
    return _format_utc(datetime.now(timezone.utc))


def _format_utc(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


__all__ = ["GUIHistoryStore", "StoredImage"]
