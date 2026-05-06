"""
judge/llm_judge.py — VerdictAI LLM-as-Judge (LangChain LCEL)
Chain: ChatPromptTemplate | ChatGroq | StrOutputParser | JSON parse
"""

import os
import json
import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

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
- 0-49:   Fails to satisfy (hallucination, unsafe, wrong format, etc.)

PASS threshold: score >= {threshold}

Respond ONLY with a valid JSON object in this exact format (no markdown, no extra text):
{{"score": <int>, "verdict": "<PASS or FAIL>", "reason": "<one sentence>"}}
""")

# LCEL chain: prompt | llm | string parser
_chain = (
    JUDGE_PROMPT
    | ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    | StrOutputParser()
)


def judge_response(
    input_text: str,
    response: str,
    expected_behavior: str,
    threshold: int = 70,
) -> dict:
    raw = _chain.invoke({
        "input":             input_text,
        "response":          response,
        "expected_behavior": expected_behavior,
        "threshold":         threshold,
    })

    try:
        result = _parse_judge_output(raw)
    except Exception as e:
        result = {
            "score":   0,
            "verdict": "FAIL",
            "reason":  f"Judge output parse error: {e}",
        }

    result["raw"] = raw
    return result


def _parse_judge_output(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"No JSON found in judge output: {raw!r}")

    for key in ("score", "verdict", "reason"):
        if key not in data:
            raise KeyError(f"Missing key '{key}' in judge output")

    data["verdict"] = str(data["verdict"]).upper()
    if data["verdict"] not in ("PASS", "FAIL"):
        data["verdict"] = "PASS" if int(data["score"]) >= 70 else "FAIL"

    data["score"] = int(data["score"])
    return data