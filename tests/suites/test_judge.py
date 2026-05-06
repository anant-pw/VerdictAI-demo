"""
tests/test_judge.py
Unit tests for LLM judge parsing logic.
No Groq API calls — tests the parser in isolation.
"""

import pytest
from judge.llm_judge import _parse_judge_output


class TestJudgeOutputParsing:

    def test_clean_json_pass(self):
        raw = '{"score": 92, "verdict": "PASS", "reason": "Correctly attributed."}'
        result = _parse_judge_output(raw)
        assert result["score"] == 92
        assert result["verdict"] == "PASS"
        assert "reason" in result

    def test_clean_json_fail(self):
        raw = '{"score": 40, "verdict": "FAIL", "reason": "Hallucinated content."}'
        result = _parse_judge_output(raw)
        assert result["score"] == 40
        assert result["verdict"] == "FAIL"

    def test_markdown_fenced_json(self):
        raw = '```json\n{"score": 85, "verdict": "PASS", "reason": "Good response."}\n```'
        result = _parse_judge_output(raw)
        assert result["score"] == 85

    def test_verdict_normalization_lowercase(self):
        raw = '{"score": 80, "verdict": "pass", "reason": "ok"}'
        result = _parse_judge_output(raw)
        assert result["verdict"] == "PASS"

    def test_verdict_fallback_from_score(self):
        # verdict is invalid string but score is high → should PASS
        raw = '{"score": 75, "verdict": "UNKNOWN", "reason": "ok"}'
        result = _parse_judge_output(raw)
        assert result["verdict"] == "PASS"

    def test_missing_key_raises(self):
        raw = '{"score": 80, "verdict": "PASS"}'  # missing "reason"
        with pytest.raises(KeyError):
            _parse_judge_output(raw)

    def test_no_json_raises(self):
        raw = "The response looks good and passes the test."
        with pytest.raises(ValueError):
            _parse_judge_output(raw)

    def test_json_embedded_in_text(self):
        raw = 'Here is my evaluation: {"score": 60, "verdict": "FAIL", "reason": "Partial."} Done.'
        result = _parse_judge_output(raw)
        assert result["score"] == 60
