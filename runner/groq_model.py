"""
groq_model.py — model-under-test caller for VerdictAI.

Returns (response_text, token_usage_dict) so runner can log tokens.
"""

from judge.groq_client import call_groq


def get_response(input_text: str, model: str = None) -> tuple[str, dict]:
    """
    Get LLM response for a test input.

    Returns:
        (response_text, {"tokens_input": int, "tokens_output": int, "tokens_total": int})
    """
    if model is None:
        from judge.groq_client import MODEL
        model = MODEL

    result = call_groq(input_text, model=model, temperature=0.7, max_tokens=512)
    return result["text"], {
        "tokens_input":  result["tokens_input"],
        "tokens_output": result["tokens_output"],
        "tokens_total":  result["tokens_total"],
    }
