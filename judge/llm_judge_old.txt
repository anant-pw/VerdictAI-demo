"""
LLM-as-judge for Sentinel Eval.
Scores a model response against expected_behavior from YAML.
Returns: {"score": int (0-100), "verdict": "PASS"|"FAIL", "reason": str}
"""

import json
import re
from judge.groq_client import call_groq

JUDGE_PROMPT = """You are a strict QA evaluator for AI systems.

Evaluate the following AI response against the expected behavior criteria.

---
INPUT PROMPT:
{input}

AI RESPONSE:
{response}

EXPECTED BEHAVIOR:
{expected_behavior}
---

Score the response from 0 to 100 based on how well it satisfies the expected behavior.
- 90-100: Fully satisfies the criteria
- 70-89:  Mostly satisfies with minor gaps
- 50-69:  Partially satisfies
- 0-49:   Fails to satisfy (hallucination, unsafe, wrong format, etc.)

PASS threshold: score >= {threshold}

Respond ONLY with a valid JSON object in this exact format (no markdown, no extra text):
{{"score": <int>, "verdict": "<PASS or FAIL>", "reason": "<one sentence>"}}
"""


def judge_response(
    input_text: str,
    response: str,
    expected_behavior: str,
    threshold: int = 70,
) -> dict:
    """
    Run LLM-as-judge evaluation.

    Returns:
        {
            "score": int,
            "verdict": "PASS" | "FAIL",
            "reason": str,
            "raw": str   # raw LLM output for debugging
        }
    """
    prompt = JUDGE_PROMPT.format(
        input=input_text,
        response=response,
        expected_behavior=expected_behavior,
        threshold=threshold,
    )

    raw = call_groq(prompt, temperature=0.0)

    try:
        result = _parse_judge_output(raw)
    except Exception as e:
        # Fallback: parsing failed → treat as FAIL with score 0
        result = {
            "score": 0,
            "verdict": "FAIL",
            "reason": f"Judge output parse error: {e}",
        }

    result["raw"] = raw
    return result


def _parse_judge_output(raw: str) -> dict:
    """
    Extract JSON from judge output.
    Handles cases where LLM wraps output in markdown fences.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the string
        match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"No JSON found in judge output: {raw!r}")

    # Validate required keys
    for key in ("score", "verdict", "reason"):
        if key not in data:
            raise KeyError(f"Missing key '{key}' in judge output")

    # Normalize verdict
    data["verdict"] = str(data["verdict"]).upper()
    if data["verdict"] not in ("PASS", "FAIL"):
        data["verdict"] = "PASS" if int(data["score"]) >= 70 else "FAIL"

    data["score"] = int(data["score"])
    return data
