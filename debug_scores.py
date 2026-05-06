# debug_scores.py
from dotenv import load_dotenv
load_dotenv("config.env")

from judge.relevance_scorer import get_relevance_score
from judge.hallucination_detector import detect_hallucination

response = "Guido van Rossum invented Python in 1991."
expected = "Answer must mention Guido van Rossum as the inventor of Python."

print("Testing relevance scorer...")
try:
    r = get_relevance_score(response, expected)
    print(f"  Relevance: {r}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nTesting hallucination detector...")
try:
    h = detect_hallucination(response, expected)
    print(f"  Hallucination: {h}")
except Exception as e:
    print(f"  ERROR: {e}")