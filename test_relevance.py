# test_relevance.py
from judge.relevance_scorer import get_relevance_score

response = "Python was invented by Guido van Rossum in 1991."
expected = "Answer must mention Guido van Rossum as the inventor of Python."

result = get_relevance_score(response, expected)
print("Relevance Score Result:")
print(f"  Score: {result['score']}")
print(f"  Verdict: {result['verdict']}")
print(f"  Cosine Similarity: {result['cosine_similarity']}")
print(f"  Model: {result['model']}")
# Expected: {"score": 85+, "verdict": "PASS"}