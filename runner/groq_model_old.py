"""
groq_model.py — replaces dummy_model.py for Layer 3+
Calls Groq to generate actual model responses for test inputs.
"""

from judge.groq_client import call_groq


def get_response(input_text: str, model: str = None) -> tuple[str, dict]:
    if model is None:
        from judge.groq_client import MODEL
        model = MODEL
    result = call_groq(input_text, model=model, temperature=0.7)
    return result["text"], {
        "tokens_input": result["tokens_input"],
        "tokens_output": result["tokens_output"],
        "tokens_total": result["tokens_total"],
    }