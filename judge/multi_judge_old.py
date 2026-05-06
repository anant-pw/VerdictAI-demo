"""
judge/multi_judge.py — VerdictAI Multi-Judge Consensus
Runs Groq + Gemini judges in parallel via asyncio.
Averages scores, flags disagreements, returns consensus verdict.
"""

import os
import asyncio
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from judge.llm_judge import _parse_judge_output

load_dotenv("config.env")

JUDGE_PROMPT = ChatPromptTemplate.from_template("""
You are a strict QA evaluator for AI systems.

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
- 0-49:   Fails to satisfy

PASS threshold: score >= {threshold}

Respond ONLY with a valid JSON object (no markdown, no extra text):
{{"score": <int>, "verdict": "<PASS or FAIL>", "reason": "<one sentence>"}}
""")

# Groq chain
_groq_chain = (
    JUDGE_PROMPT
    | ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    | StrOutputParser()
)

# Gemini chain
# _gemini_chain = (
#     JUDGE_PROMPT
#     | ChatGoogleGenerativeAI(
#         model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
#         temperature=0.0,
#         google_api_key=os.getenv("GEMINI_API_KEY"),
#     )
#     | StrOutputParser()
# )

# SambaNova chain
_gemini_chain = (
    JUDGE_PROMPT
    | ChatOpenAI(
        model=os.getenv("SAMBANOVA_MODEL", "Meta-Llama-3.1-8B-Instruct"),
        temperature=0.0,
        api_key=os.getenv("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    | StrOutputParser()
)




async def _call_judge(chain, inputs: dict, judge_name: str) -> dict:
    """Run a single judge chain asynchronously."""
    try:
        raw = await chain.ainvoke(inputs)
        result = _parse_judge_output(raw)
        result["judge"] = judge_name
        result["raw"] = raw
        return result
    except Exception as e:
        return {
            "judge":   judge_name,
            "score":   0,
            "verdict": "FAIL",
            "reason":  f"Judge error: {e}",
            "raw":     "",
        }


async def _run_both_judges(inputs: dict) -> tuple[dict, dict]:
    """Fire both judges in parallel."""
    groq_result, gemini_result = await asyncio.gather(
        _call_judge(_groq_chain, inputs, "groq"),
        _call_judge(_gemini_chain, inputs, "gemini"),
    )
    return groq_result, gemini_result


def multi_judge_response(
    input_text: str,
    response: str,
    expected_behavior: str,
    threshold: int = 70,
    disagreement_threshold: int = 20,
) -> dict:
    """
    Run Groq + Gemini judges in parallel, return consensus result.

    Args:
        input_text:              Test input prompt
        response:                Model response to evaluate
        expected_behavior:       Expected behavior criteria from YAML
        threshold:               Score threshold for PASS (default 70)
        disagreement_threshold:  Score gap that triggers disagreement flag (default 20)

    Returns:
        {
            "score":        int   (average of both judges),
            "verdict":      "PASS" | "FAIL",
            "reason":       str   (combined reason),
            "groq":         dict  (individual groq result),
            "gemini":       dict  (individual gemini result),
            "disagreement": bool  (True if scores differ by > threshold),
            "raw":          str
        }
    """
    inputs = {
        "input":             input_text,
        "response":          response,
        "expected_behavior": expected_behavior,
        "threshold":         threshold,
    }

    # Safe async execution for Python 3.10+
    import asyncio as _asyncio
    try:
        _loop = _asyncio.get_running_loop()
        # Already inside async context (e.g. Jupyter/Streamlit)
        try:
            import nest_asyncio as _nest
            _nest.apply()
        except ImportError:
            pass
        groq_result, gemini_result = _loop.run_until_complete(_run_both_judges(inputs))
    except RuntimeError:
        groq_result, gemini_result = _asyncio.run(_run_both_judges(inputs))

    # Consensus scoring
    #avg_score = int((groq_result["score"] + gemini_result["score"]) / 2)
    valid_results = [
        r for r in [groq_result, gemini_result]
        if not str(r.get("reason", "")).startswith("Judge error:")
    ]
    if not valid_results:
        # both failed — return conservative FAIL
        avg_score = 0
    elif len(valid_results) == 1:
        # only one judge succeeded — use it directly, flag it
        avg_score = valid_results[0]["score"]
        print(f"   ⚠️  Only one judge succeeded — using {valid_results[0]['judge']} score alone")
    else:
        avg_score = int(sum(r["score"] for r in valid_results) / len(valid_results))

    score_gap = abs(groq_result["score"] - gemini_result["score"])
    disagreement = score_gap > disagreement_threshold

    verdict = "PASS" if avg_score >= threshold else "FAIL"

    reason = (
        f"Groq: {groq_result['reason']} | "
        f"Gemini: {gemini_result['reason']}"
    )

    if disagreement:
        print(
            f"   ⚠️  JUDGE DISAGREEMENT: Groq={groq_result['score']} "
            f"Gemini={gemini_result['score']} (gap={score_gap})"
        )

    return {
        "score":        avg_score,
        "verdict":      verdict,
        "reason":       reason,
        "groq":         groq_result,
        "gemini":       gemini_result,
        "disagreement": disagreement,
        "raw":          f"groq={groq_result['raw']} | gemini={gemini_result['raw']}",
    }
