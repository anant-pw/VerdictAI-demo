# reports/allure_writer.py
"""
Generate Allure JSON results without pytest.
Each test case gets 5 steps in the Allure timeline:
  1. Get LLM Response
  2. Run Heuristic Assertions
  3. LLM Judge (Groq + Cerebras)
  4. Relevance Score      (full mode only — N/A for judge_only)
  5. Hallucination Score  (full mode only — N/A for judge_only)
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class AllureWriter:
    """Write Allure-compatible JSON results."""

    def __init__(self, output_dir: str = "allure-results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def write_test_result(self, test_data: Dict):
        test_id  = test_data.get("id", "unknown")
        verdict  = test_data.get("verdict", {})

        # verdict is a dict {"verdict": "PASS/FAIL", "reason": "..."}
        verdict_str    = verdict.get("verdict", "FAIL") if isinstance(verdict, dict) else str(verdict)
        verdict_reason = verdict.get("reason", "")      if isinstance(verdict, dict) else ""

        status     = "passed" if verdict_str == "PASS" else "failed"
        test_uuid  = str(uuid.uuid4())
        now_ms     = int(datetime.now().timestamp() * 1000)

        result = {
            "uuid":        test_uuid,
            "historyId":   test_id,
            "name":        f"Test: {test_id}",
            "fullName":    f"verdictai.{test_id}",
            "status":      status,
            "statusDetails": {
                "message": verdict_reason if status == "failed" else "",
                "trace":   ""
            },
            "start":  now_ms,
            "stop":   now_ms,
            "labels": [
                {"name": "suite",     "value": "VerdictAI"},
                {"name": "severity",  "value": "critical" if status == "failed" else "normal"},
                {"name": "framework", "value": "verdictai"},
            ],
            "parameters":  [],
            "attachments": self._create_attachments(test_data, test_uuid),
            "steps":       self._create_steps(test_data, verdict_str, verdict_reason),
        }

        result_file = self.output_dir / f"{test_uuid}-result.json"
        result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ── Steps ──────────────────────────────────────────────────────────────

    def _create_steps(self, test_data: Dict, verdict_str: str, verdict_reason: str) -> List[Dict]:
        steps = []
        now   = int(datetime.now().timestamp() * 1000)

        # ── Step 1: LLM Response ─────────────────────────────────────────
        steps.append({
            "name":   f"Get LLM Response ({test_data.get('latency_ms', 0)}ms)",
            "status": "passed",
            "start":  now,
            "stop":   now,
        })

        # ── Step 2: Heuristic Assertions ─────────────────────────────────
        heuristic_pass    = test_data.get("heuristic_pass", True)
        heuristic_results = test_data.get("heuristic_results", [])

        h_steps = []
        for r in (heuristic_results or []):
            h_steps.append({
                "name":   f"{r.get('type','assertion')} — value: {r.get('value','')}",
                "status": "passed" if r.get("passed") else "failed",
                "start":  now,
                "stop":   now,
            })

        failed_assertions = [
            f"{r.get('type')}({r.get('value')}) FAILED"
            for r in (heuristic_results or []) if not r.get("passed")
        ]

        steps.append({
            "name":   f"Run Heuristic Assertions ({'PASS' if heuristic_pass else 'FAIL'})",
            "status": "passed" if heuristic_pass else "failed",
            "start":  now,
            "stop":   now,
            "steps":  h_steps,
            "statusDetails": {
                "message": " | ".join(failed_assertions),
                "trace":   ""
            },
        })

        # ── Step 3: Multi-Judge Score ─────────────────────────────────────
        judge = test_data.get("judge")
        if judge:
            judge_score   = judge.get("score", "N/A")
            judge_verdict = judge.get("verdict", "FAIL")
            groq_score    = judge.get("groq",     {}).get("score", "N/A")
            groq_reason   = judge.get("groq",     {}).get("reason", "")
            second        = judge.get("cerebras") or judge.get("sambanova") or judge.get("gemini") or {}
            second_score  = second.get("score",  "N/A")
            second_name   = second.get("judge",  "Judge-2").capitalize()
            second_reason = second.get("reason", "")
            disagree      = judge.get("disagreement", False)

            step_name = f"LLM Judge (Score: {judge_score})"
            if disagree:
                step_name += f"  ⚠️ DISAGREEMENT Groq={groq_score} {second_name}={second_score}"

            steps.append({
                "name":   step_name,
                "status": "passed" if judge_verdict == "PASS" else "failed",
                "start":  now,
                "stop":   now,
                "statusDetails": {
                    "message": (
                        f"Groq ({groq_score}): {groq_reason}\n"
                        f"{second_name} ({second_score}): {second_reason}"
                    ),
                    "trace": ""
                },
            })
        else:
            steps.append({
                "name":   "LLM Judge — skipped (API error / no judge data)",
                "status": "broken",
                "start":  now,
                "stop":   now,
            })

        # ── Step 4: Relevance Score ───────────────────────────────────────
        relevance = test_data.get("relevance")
        if relevance is not None:
            rel_score  = relevance.get("score", 0)
            rel_cosine = relevance.get("cosine_similarity", 0)
            rel_pass   = rel_score >= 40   # matches min_relevance threshold
            steps.append({
                "name":   f"Relevance Score: {rel_score:.1f} (cosine={rel_cosine:.3f})",
                "status": "passed" if rel_pass else "failed",
                "start":  now,
                "stop":   now,
                "statusDetails": {
                    "message": "" if rel_pass else f"Relevance {rel_score:.1f} below threshold 40",
                    "trace":   ""
                },
            })
        else:
            steps.append({
                "name":   "Relevance Score — N/A (judge_only suite)",
                "status": "passed",   # not a failure — intentionally skipped
                "start":  now,
                "stop":   now,
            })

        # ── Step 5: Hallucination Score ───────────────────────────────────
        hallucination = test_data.get("hallucination")
        if hallucination is not None:
            hal_score     = hallucination.get("score", 0)
            hal_supported = hallucination.get("claims_supported", 0)
            hal_total     = hallucination.get("claims_total", 0)
            hal_pass      = hal_score >= 30   # matches (100 - max_hallucination=70) threshold
            steps.append({
                "name":   f"Hallucination Score: {hal_score:.1f}% ({hal_supported}/{hal_total} claims supported)",
                "status": "passed" if hal_pass else "failed",
                "start":  now,
                "stop":   now,
                "statusDetails": {
                    "message": "" if hal_pass else f"Only {hal_supported}/{hal_total} claims supported — possible hallucination",
                    "trace":   ""
                },
            })
        else:
            steps.append({
                "name":   "Hallucination Score — N/A (judge_only suite)",
                "status": "passed",   # not a failure — intentionally skipped
                "start":  now,
                "stop":   now,
            })

        # ── Final Verdict step ────────────────────────────────────────────
        steps.append({
            "name":   f"Final Verdict: {verdict_str}",
            "status": "passed" if verdict_str == "PASS" else "failed",
            "start":  now,
            "stop":   now,
            "statusDetails": {
                "message": verdict_reason if verdict_str == "FAIL" else "",
                "trace":   ""
            },
        })

        return steps

    # ── Attachments ────────────────────────────────────────────────────────

    def _create_attachments(self, test_data: Dict, test_uuid: str) -> List[Dict]:
        attachments = []

        def _attach(name, content, ext="txt"):
            text = str(content) if content is not None else "(empty)"
            if not text.strip():
                text = "(empty)"
            fname = f"{test_uuid}-{name.replace(' ', '_')}.{ext}"
            (self.output_dir / fname).write_text(text, encoding="utf-8")
            attachments.append({"name": name, "source": fname, "type": "text/plain"})

        _attach("Input_Prompt",  test_data.get("input")  or "(no input recorded)")
        _attach("LLM_Response",  test_data.get("response") or "(no response recorded)")

        judge = test_data.get("judge")
        if judge:
            second       = judge.get("cerebras") or judge.get("sambanova") or judge.get("gemini") or {}
            second_name  = second.get("judge", "Judge-2").capitalize()
            judge_text   = (
                f"Score:   {judge.get('score')}\n"
                f"Verdict: {judge.get('verdict')}\n"
                f"Reason:  {judge.get('reason', 'N/A')}\n\n"
                f"Groq ({judge.get('groq', {}).get('score', 'N/A')}): "
                f"{judge.get('groq', {}).get('reason', '')}\n"
                f"{second_name} ({second.get('score', 'N/A')}): "
                f"{second.get('reason', '')}\n"
                f"Disagreement: {judge.get('disagreement', False)}"
            )
            _attach("Judge Result", judge_text)

        relevance = test_data.get("relevance")
        if relevance:
            _attach("Relevance Detail",
                    f"Score: {relevance.get('score')}\nCosine: {relevance.get('cosine_similarity')}")

        hallucination = test_data.get("hallucination")
        if hallucination:
            hal_text = (
                f"Score:       {hallucination.get('score')}%\n"
                f"Supported:   {hallucination.get('claims_supported')}/{hallucination.get('claims_total')}\n"
                f"Contradicted:{hallucination.get('claims_contradicted', 0)}\n"
            )
            _attach("Hallucination Detail", hal_text)

        verdict = test_data.get("verdict", {})
        if isinstance(verdict, dict) and verdict.get("verdict") == "FAIL":
            _attach("Failure Reason", verdict.get("reason", "Unknown"))

        return attachments

    # ── Environment + Categories ───────────────────────────────────────────

    def write_environment(self, env_data: Dict):
        env_file = self.output_dir / "environment.properties"
        lines = [f"{k}={v}" for k, v in env_data.items()]
        env_file.write_text("\n".join(lines), encoding="utf-8")

    def write_categories(self):
        categories = [
            {
                "name": "Heuristic Failures",
                "matchedStatuses": ["failed"],
                "messageRegex": ".*Heuristic.*|.*contains.*|.*max_length.*"
            },
            {
                "name": "Judge Failures",
                "matchedStatuses": ["failed"],
                "messageRegex": ".*judge.*|.*LLM judge.*"
            },
            {
                "name": "Relevance Failures",
                "matchedStatuses": ["failed"],
                "messageRegex": ".*[Rr]elevance.*below.*"
            },
            {
                "name": "Hallucination Failures",
                "matchedStatuses": ["failed"],
                "messageRegex": ".*claims supported.*|.*hallucination.*"
            },
            {
                "name": "API / Infrastructure Failures",
                "matchedStatuses": ["broken"],
                "messageRegex": ".*API error.*|.*no judge.*|.*429.*"
            },
        ]
        cat_file = self.output_dir / "categories.json"
        cat_file.write_text(json.dumps(categories, indent=2), encoding="utf-8")
