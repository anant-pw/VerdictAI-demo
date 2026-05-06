"""
main.py — VerdictAI entry point (Layer 3)
Usage: python -m runner.main [--suite path/to/suite.yaml] [--no-judge]
"""

import argparse
import sys
from runner.runner import run_suite


def _get_verdict_label(verdict) -> str:
    """Safely extract verdict string whether it's a dict or plain string."""
    if isinstance(verdict, dict):
        return verdict.get("verdict", "FAIL")
    return str(verdict) if verdict else "FAIL"


def main():
    parser = argparse.ArgumentParser(description="VerdictAI — LLM Testing Framework")
    parser.add_argument(
        "--suite",
        default="tests/suites/hallucination.yaml",
        help="Path to YAML test suite",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge (heuristics only)",
    )
    args = parser.parse_args()

    print(f"\n🚀 VerdictAI — Running suite: {args.suite}")
    print(f"   LLM Judge: {'DISABLED' if args.no_judge else 'ENABLED'}")

    results = run_suite(args.suite, use_judge=not args.no_judge)

    # Summary — verdict is a dict {"verdict": "PASS/FAIL", "reason": ...}
    total = len(results)
    passed = sum(1 for r in results if _get_verdict_label(r["verdict"]) == "PASS")
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} passed | {failed} failed")

    if failed > 0:
        print("FAILED TESTS:")
        for r in results:
            if _get_verdict_label(r["verdict"]) == "FAIL":
                judge_score = r["judge"]["score"] if r.get("judge") else "N/A"
                reason = r["verdict"].get("reason", "") if isinstance(r["verdict"], dict) else ""
                print(f"  ❌ {r['id']} (judge={judge_score} | reason: {reason})")
    # Exit 0 always — let GitHub Actions workflow control job failure
    # Individual suite failures are visible in step logs and uploaded DB artifacts
    # Change to sys.exit(1) if you want the whole CI job to fail on any test failure

    sys.exit(0)


if __name__ == "__main__":
    main()