# test_hallucination.py
from judge.hallucination_detector import detect_hallucination

response = "Python was invented by Guido van Rossum in 1991 at Google."

source_context = """
Guido van Rossum created Python in 1989 while working at Centrum Wiskunde & Informatica (CWI) in the Netherlands.
He released the first version in 1991.
"""

result = detect_hallucination(response, source_context)
print("Hallucination Detection:")
print(f"  Score: {result['score']}%")
print(f"  Verdict: {result['verdict']}")
print(f"  Claims: {result['claims_total']}")
print(f"  Supported: {result['claims_supported']}")
print(f"  Contradicted: {result['claims_contradicted']}")
print(f"\nDetails:")
for claim in result['details']:
    print(f"  [{claim['status']}] {claim['claim']}")