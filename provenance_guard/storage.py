"""SQLite persistence: content records + a structured, append-only audit log.

Two tables:
  content    — current state of each submission (one row per content_id)
  audit_log  — append-only event stream (classification + appeal events)

SQLite is built into Python; no external setup. JSON columns hold signal
breakdowns so the log stays structured but flexible.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from . import config

_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS content (
                content_id     TEXT PRIMARY KEY,
                creator_id     TEXT NOT NULL,
                text           TEXT NOT NULL,
                attribution    TEXT NOT NULL,
                confidence     REAL NOT NULL,
                ai_likelihood  REAL NOT NULL,
                llm_score      REAL,
                stylo_score    REAL,
                status         TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type     TEXT NOT NULL,        -- classification | appeal
                content_id     TEXT NOT NULL,
                creator_id     TEXT,
                timestamp      TEXT NOT NULL,
                attribution    TEXT,
                confidence     REAL,
                ai_likelihood  REAL,
                llm_score      REAL,
                stylo_score    REAL,
                status         TEXT,
                appeal_reasoning TEXT,
                details        TEXT                  -- JSON blob (signals, etc.)
            );
            """
        )


def record_classification(creator_id: str, text: str, result: dict) -> str:
    """Persist a new classification and append an audit entry. Returns content_id."""
    content_id = str(uuid.uuid4())
    ts = _now_iso()
    llm_score = result["signals"]["llm"]["score"]
    stylo_score = result["signals"]["stylometric"]["score"]

    with _lock, _connect() as conn:
        conn.execute(
            """INSERT INTO content (content_id, creator_id, text, attribution,
                   confidence, ai_likelihood, llm_score, stylo_score, status,
                   created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                content_id, creator_id, text, result["attribution"],
                result["confidence"], result["ai_likelihood"], llm_score,
                stylo_score, "classified", ts, ts,
            ),
        )
        conn.execute(
            """INSERT INTO audit_log (event_type, content_id, creator_id,
                   timestamp, attribution, confidence, ai_likelihood, llm_score,
                   stylo_score, status, appeal_reasoning, details)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "classification", content_id, creator_id, ts,
                result["attribution"], result["confidence"],
                result["ai_likelihood"], llm_score, stylo_score, "classified",
                None, json.dumps({
                    "signals": result["signals"],
                    "degraded_mode": result["degraded_mode"],
                    "label_variant": result["label"]["variant"],
                }),
            ),
        )
    return content_id


def get_content(content_id: str) -> dict | None:
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM content WHERE content_id = ?", (content_id,)
        ).fetchone()
    return dict(row) if row else None


def record_appeal(content_id: str, creator_reasoning: str) -> dict | None:
    """Set status -> under_review and log the appeal with the original decision.

    Returns the updated content dict, or None if content_id is unknown.
    """
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM content WHERE content_id = ?", (content_id,)
        ).fetchone()
        if row is None:
            return None
        content = dict(row)
        ts = _now_iso()

        conn.execute(
            "UPDATE content SET status = ?, updated_at = ? WHERE content_id = ?",
            ("under_review", ts, content_id),
        )
        conn.execute(
            """INSERT INTO audit_log (event_type, content_id, creator_id,
                   timestamp, attribution, confidence, ai_likelihood, llm_score,
                   stylo_score, status, appeal_reasoning, details)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "appeal", content_id, content["creator_id"], ts,
                content["attribution"], content["confidence"],
                content["ai_likelihood"], content["llm_score"],
                content["stylo_score"], "under_review", creator_reasoning,
                json.dumps({
                    "original_decision": {
                        "attribution": content["attribution"],
                        "confidence": content["confidence"],
                        "ai_likelihood": content["ai_likelihood"],
                        "llm_score": content["llm_score"],
                        "stylo_score": content["stylo_score"],
                        "classified_at": content["created_at"],
                    }
                }),
            ),
        )
        content["status"] = "under_review"
        content["updated_at"] = ts
    return content


def get_log(limit: int = 50) -> list[dict]:
    """Return the most recent audit entries (newest first), JSON-ready."""
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    entries = []
    for r in rows:
        e = dict(r)
        if e.get("details"):
            try:
                e["details"] = json.loads(e["details"])
            except (json.JSONDecodeError, TypeError):
                pass
        entries.append(e)
    return entries


def get_review_queue() -> list[dict]:
    """Content currently under_review — the human reviewer's appeal queue."""
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM content WHERE status = 'under_review' "
            "ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
