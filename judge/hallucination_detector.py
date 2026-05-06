# judge/hallucination_detector.py
"""
Hallucination detection via claim extraction and batched verification.

TOKEN OPTIMISATIONS (v2):
  1. Batched verification  — all claims verified in ONE Groq call (was N calls)
  2. Tighter prompts       — extraction prompt trimmed ~40%
  3. max_tokens capped     — extraction=200, verification=400
  4. Fallback              — if batch parse fails, falls back to sequential
"""

from typing import Dict, List
import json
from groq import Groq
import os


class HallucinationDetector:
    """Detect hallucinated claims in LLM responses."""

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv("config.env")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"

    def detect(self, response: str, source_context: str) -> Dict:
        """
        Detect hallucinations by comparing claims to source.
        Returns dict with score (0-100, higher = less hallucination).
        """
        claims = self._extract_claims(response)

        if not claims:
            return {
                "score": 100,
                "verdict": "PASS",
                "claims_total": 0,
                "claims_supported": 0,
                "claims_contradicted": 0,
                "claims_unsupported": 0,
                "details": [],
            }

        verified     = self._verify_claims_batched(claims, source_context)
        supported    = sum(1 for c in verified if c["status"] == "SUPPORTED")
        contradicted = sum(1 for c in verified if c["status"] == "CONTRADICTED")
        unsupported  = sum(1 for c in verified if c["status"] == "UNSUPPORTED")
        total        = len(verified)
        support_rate = (supported / total * 100) if total > 0 else 100

        return {
            "score":               round(support_rate, 2),
            "verdict":             "PASS" if support_rate >= 80 else "FAIL",
            "claims_total":        total,
            "claims_supported":    supported,
            "claims_contradicted": contradicted,
            "claims_unsupported":  unsupported,
            "details":             verified,
        }

    def _extract_claims(self, response: str) -> List[str]:
        """Tightened prompt (~45 tokens), capped output at 200 tokens."""
        prompt = (
            "Extract factual claims from the text below. "
            'Return ONLY a JSON array of strings, e.g. ["claim1","claim2"].\n\n'
            f"Text: {response}\n\nJSON array:"
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            content = completion.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].lstrip("json").strip()
            claims = json.loads(content)
            return claims if isinstance(claims, list) else []
        except Exception as e:
            print(f"[WARN] Claim extraction failed: {e}")
            return []

    def _verify_claims_batched(self, claims: List[str], source_context: str) -> List[Dict]:
        """
        All claims verified in ONE call instead of N calls.
        Was: N x ~200 tokens. Now: 1 x ~(200 + 20xN) tokens.
        Falls back to sequential on JSON parse failure.
        """
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        prompt = (
            "Check each claim against the source. "
            'Reply ONLY with a JSON array: [{"claim":"...","status":"SUPPORTED|CONTRADICTED|UNSUPPORTED"}]\n\n'
            "SUPPORTED=stated/implied in source | CONTRADICTED=source disagrees | UNSUPPORTED=not mentioned\n\n"
            f"Source:\n{source_context}\n\n"
            f"Claims:\n{numbered}\n\nJSON array:"
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
            )
            content = completion.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].lstrip("json").strip()
            parsed = json.loads(content)
            verified = []
            for i, item in enumerate(parsed):
                status = str(item.get("status", "UNSUPPORTED")).upper().strip()
                if status not in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED"):
                    status = "UNSUPPORTED"
                verified.append({
                    "claim":  item.get("claim", claims[i] if i < len(claims) else ""),
                    "status": status,
                })
            return verified
        except Exception as e:
            print(f"[WARN] Batched verification failed ({e}), falling back to sequential")
            return self._verify_claims_sequential(claims, source_context)

    def _verify_claims_sequential(self, claims: List[str], source_context: str) -> List[Dict]:
        """Fallback: one call per claim."""
        verified = []
        for claim in claims:
            prompt = (
                f"Source:\n{source_context}\n\n"
                f"Claim: {claim}\n\n"
                "One word — SUPPORTED, CONTRADICTED, or UNSUPPORTED:"
            )
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10,
                )
                result = completion.choices[0].message.content.strip().upper()
                status = result if result in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED") else "UNSUPPORTED"
            except Exception as e:
                print(f"[WARN] Single claim verification failed: {e}")
                status = "UNSUPPORTED"
            verified.append({"claim": claim, "status": status})
        return verified


_detector = None

def detect_hallucination(response: str, source_context: str) -> Dict:
    """Lazy-loaded singleton entry point."""
    global _detector
    if _detector is None:
        _detector = HallucinationDetector()
    return _detector.detect(response, source_context)
