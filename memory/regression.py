"""
memory/regression.py — VerdictAI Regression Eval
Compares current score vs last run for same test_id.
Flags if score dropped > threshold (default 10 points).
"""

#from memory.store import get_history


def check_regression(test_id: str, current_score: int, drop_threshold: int = 10) -> dict:
    """
    Compare current score against last stored score for test_id.

    Returns:
        {
            "regressed":   bool,
            "prev_score":  int | None,
            "drop":        int | None,
            "message":     str
        }
    """
    from memory.store import get_history
    history = get_history(test_id, limit=2)

    # Need at least 1 previous run to compare
    if len(history) < 2:
        return {
            "regressed": False,
            "prev_score": None,
            "drop": None,
            "message": "No previous run to compare.",
        }

    # history[0] = current run (just saved), history[1] = previous
    prev_score = history[1].get("score")

    if prev_score is None:
        return {
            "regressed": False,
            "prev_score": None,
            "drop": None,
            "message": "Previous run had no score.",
        }

    drop = prev_score - current_score

    if drop > drop_threshold:
        return {
            "regressed": True,
            "prev_score": prev_score,
            "drop": drop,
            "message": f"⚠️  Regression: score dropped {drop} pts ({prev_score} → {current_score})",
        }

    return {
        "regressed": False,
        "prev_score": prev_score,
        "drop": drop,
        "message": f"✅ No regression ({prev_score} → {current_score})",
    }