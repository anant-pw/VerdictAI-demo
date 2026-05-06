"""
memory/store.py — Layer 4
Stores every test result to SQLite for trend detection and self-healing.
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.getenv("SENTINEL_DB", "memory/sentinel.db")


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ✅ SAFE SERIALIZER (critical)
def _serialize(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def init_db():
    """Create table if not exists."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TEXT    NOT NULL,
                suite_name    TEXT    NOT NULL,
                test_id       TEXT    NOT NULL,
                verdict       TEXT    NOT NULL,
                score         INTEGER,
                reason        TEXT,
                response      TEXT,
                latency_ms    INTEGER,
                regressed     INTEGER DEFAULT 0,
                score_drop    INTEGER
            )
        """)
        conn.commit()


def save_result(suite_name: str, result: dict):
    """Append one test result to the DB."""

    judge = result.get("judge") or {}
    regression = result.get("regression") or {}

    # ✅ FIX: handle structured verdict
    verdict = result.get("verdict")
    if isinstance(verdict, dict):
        verdict_label = verdict.get("verdict")
        verdict_reason = verdict.get("reason")
    else:
        verdict_label = verdict
        verdict_reason = judge.get("reason")

    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO results
                (run_timestamp, suite_name, test_id, verdict, score, reason, response, latency_ms, regressed, score_drop)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            suite_name,
            result["id"],
            _serialize(verdict_label),                 # ✅ always safe
            _serialize(judge.get("score")),
            _serialize(verdict_reason),                # ✅ use final verdict reason
            _serialize(result.get("response", "")[:2000]),
            _serialize(result.get("latency_ms")),
            int(regression.get("regressed", False)),
            _serialize(regression.get("drop")),
        ))
        conn.commit()

    result["run_timestamp"] = datetime.utcnow().isoformat()

def get_history(test_id: str, limit: int = 10) -> list[dict]:
    """Return last N results for a test_id, newest first."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT score, verdict, run_timestamp
            FROM results
            WHERE test_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (test_id, limit)).fetchall()
    return [dict(r) for r in rows]

def get_consecutive_failures(test_id: str, n: int = 3) -> int:
    """Return the count of consecutive FAIL results for test_id (most recent first, up to n)."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT verdict FROM results
            WHERE test_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (test_id, n)).fetchall()
    count = 0
    for row in rows:
        if str(row["verdict"]).upper() == "FAIL":
            count += 1
        else:
            break
    return count
