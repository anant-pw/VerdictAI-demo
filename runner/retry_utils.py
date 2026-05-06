"""
runner/retry_utils.py — VerdictAI retry + backoff utility

Wraps API calls with exponential backoff on rate limit (429) errors.
Uses tenacity library. Handles Groq, SambaNova, and Gemini 429s.
"""

import time
import logging
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

logger = logging.getLogger("verdictai.retry")

# ── Rate limit detection ──────────────────────────────────────────────────────

def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception is a 429 / rate limit error from any provider."""
    msg = str(exc).lower()
    return any(keyword in msg for keyword in [
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "resource_exhausted",   # Gemini
        "ratelimitexceeded",    # SambaNova
    ])


def _is_retryable(exc: Exception) -> bool:
    """Retry on rate limits and transient network errors."""
    msg = str(exc).lower()
    return _is_rate_limit_error(exc) or any(k in msg for k in [
        "connection", "timeout", "server error", "503", "502"
    ])


# ── Tenacity retry decorator ─────────────────────────────────────────────────

def with_retry(max_attempts: int = 4, min_wait: int = 5, max_wait: int = 60):
    """
    Decorator factory for retrying API calls with exponential backoff.

    Args:
        max_attempts: Total attempts before giving up (default 4)
        min_wait:     Minimum wait in seconds between retries (default 5)
        max_wait:     Maximum wait in seconds between retries (default 60)

    Usage:
        @with_retry()
        def call_my_api(...):
            ...
    """
    return retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=min_wait, max=max_wait),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# ── Inter-case sleep ──────────────────────────────────────────────────────────

def inter_case_sleep(seconds: float = 2.0):
    """
    Sleep between test cases to avoid hitting per-minute rate limits.
    Set seconds=0 to disable.
    """
    if seconds > 0:
        print(f"   ⏳ Rate-limit buffer: sleeping {seconds}s before next case...")
        time.sleep(seconds)
