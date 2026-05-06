# runner/assertions.py

def run_assertions(response: str, assertions: list[dict]) -> list[dict]:
    results = []
    for a in assertions:
        t = a["type"]
        val = a.get("value", "")
        if t == "contains":
            passed = val.lower() in response.lower()
        elif t == "not_contains":
            passed = val.lower() not in response.lower()
        elif t == "max_length":
            passed = len(response) <= int(val)
        elif t == "min_length":
            passed = len(response) >= int(val)
        else:
            passed = False
        results.append({"type": t, "value": val, "passed": passed})
    return results