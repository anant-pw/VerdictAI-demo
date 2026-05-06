# judge/judge_cache.py
"""
Judge response cache — SQLite-backed, zero extra dependencies.

TOKEN OPTIMISATION (v2):
  Caches judge results keyed on MD5(input+response+expected_behavior).
  Cache hit = 0 Groq calls, 0 tokens.  Especially useful during
  iterative development when the same suite is run repeatedly.

Usage:
    from judge.judge_cache import get_cached, set_cached
    key = cache_key(input_text, response, expected_behavior)
    hit = get_cached(key)
    if hit:
        return hit
    result = run_judge(...)
    set_cached(key, result)
"""

import hashlib
import json
import sqlite3
import os
from datetime import datetime
from pathlib import Path

_DB = os.getenv("VERDICTAI_DB", "verdictai.db")
_TTL_HOURS = 24          # cache entries expire after 24 hours
_ENABLED = os.getenv("VERDICTAI_JUDGE_CACHE", "1") == "1"  # set to "0" to disable


def cache_key(input_text: str, response: str, expected_behavior: str) -> str:
    """Stable MD5 key for a (input, response, expected) triple."""
    raw = f"{input_text.strip()}||{response.strip()}||{expected_behavior.strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS judge_cache (
            cache_key   TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_judge_cache_created
        ON judge_cache(created_at)
    """)
    conn.commit()
    return conn


def get_cached(key: str) -> dict | None:
    """Return cached judge result or None if not found / expired."""
    if not _ENABLED:
        return None
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT result_json, created_at FROM judge_cache WHERE cache_key = ?",
                (key,)
            ).fetchone()
            if not row:
                return None
            # TTL check
            created = datetime.fromisoformat(row["created_at"])
            age_hours = (datetime.utcnow() - created).total_seconds() / 3600
            if age_hours > _TTL_HOURS:
                conn.execute("DELETE FROM judge_cache WHERE cache_key = ?", (key,))
                conn.commit()
                return None
            result = json.loads(row["result_json"])
            result["_cache_hit"] = True
            return result
    except Exception as e:
        print(f"[WARN] Judge cache read failed: {e}")
        return None


def set_cached(key: str, result: dict) -> None:
    """Store judge result in cache."""
    if not _ENABLED:
        return
    try:
        # Don't cache error results
        if str(result.get("reason", "")).startswith("Judge error:"):
            return
        payload = {k: v for k, v in result.items() if k != "_cache_hit"}
        with _get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO judge_cache (cache_key, result_json, created_at)
                VALUES (?, ?, ?)
            """, (key, json.dumps(payload), datetime.utcnow().isoformat()))
            conn.commit()
    except Exception as e:
        print(f"[WARN] Judge cache write failed: {e}")


def clear_cache() -> int:
    """Clear all cache entries. Returns count deleted."""
    try:
        with _get_conn() as conn:
            n = conn.execute("SELECT COUNT(*) FROM judge_cache").fetchone()[0]
            conn.execute("DELETE FROM judge_cache")
            conn.commit()
            return n
    except Exception:
        return 0


def cache_stats() -> dict:
    """Return cache hit/miss stats for the current DB."""
    try:
        with _get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM judge_cache").fetchone()[0]
            return {"cached_entries": count, "ttl_hours": _TTL_HOURS, "enabled": _ENABLED}
    except Exception:
        return {"cached_entries": 0, "ttl_hours": _TTL_HOURS, "enabled": _ENABLED}
