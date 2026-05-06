"""
Groq client wrapper for Sentinel Eval.
Handles model calls with basic retry logic.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv("config.env")
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
print(f"[DEBUG] GROQ_MODEL loaded: {MODEL}")

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in environment.")
        _client = Groq(api_key=api_key)
    return _client


def call_groq(prompt: str, model: str = MODEL, temperature: float = 0.0) -> dict:
    """
    Send a prompt to Groq and return response text + token usage.
    """
    client = get_client()
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = completion.usage  # has prompt_tokens, completion_tokens, total_tokens
    return {
        "text": completion.choices[0].message.content.strip(),
        "tokens_input": usage.prompt_tokens,
        "tokens_output": usage.completion_tokens,
        "tokens_total": usage.total_tokens,
    }
