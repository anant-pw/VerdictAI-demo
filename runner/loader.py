# runner/loader.py
import yaml
from pathlib import Path

def load_suite(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        return data.get("test_cases", [])
    return data