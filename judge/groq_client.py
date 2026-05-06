"""
Groq client wrapper for VerdictAI.

TOKEN OPTIMISATIONS (v2):
  - max_tokens=512 on model-under-test calls (was unbounded)
  - Returns token usage dict so runner can log to llm_calls table
"""

import os
from dotenv import load_dotenv
from groq import Groq
from runner.retry_utils import with_retry

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


@with_retry(max_attempts=4, min_wait=5, max_wait=60)
def call_groq(prompt: str, model: str = MODEL, temperature: float = 0.0,
              max_tokens: int = 512) -> dict:
    """
    Send a prompt to Groq. Returns dict with text + token usage.

    Args:
        prompt:      User message
        model:       Groq model ID
        temperature: Sampling temperature
        max_tokens:  Hard cap on output tokens (OPTIMISATION 5)

    Returns:
        {
            "text":          str,
            "tokens_input":  int,
            "tokens_output": int,
            "tokens_total":  int,
        }
    """
    client = get_client()
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = completion.usage
    return {
        "text":          completion.choices[0].message.content.strip(),
        "tokens_input":  usage.prompt_tokens,
        "tokens_output": usage.completion_tokens,
        "tokens_total":  usage.total_tokens,
    }
