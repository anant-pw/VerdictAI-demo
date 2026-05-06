# judge/hallucination_detector.py
"""
Hallucination detection via claim extraction and verification.
Identifies unsupported factual statements in LLM responses.
"""

from typing import Dict, List
import json
from groq import Groq
import os


class HallucinationDetector:
    """Detect hallucinated claims in LLM responses."""
    
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv('config.env')
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
    
    def detect(self, response: str, source_context: str) -> Dict:
        """
        Detect hallucinations by comparing claims to source.
        
        Args:
            response: LLM's response to check
            source_context: Ground truth / reference material
            
        Returns:
            Dict with hallucination score and claim details
        """
        # Step 1: Extract claims from response
        claims = self._extract_claims(response)
        
        if not claims:
            return {
                "score": 100,
                "verdict": "PASS",
                "claims_total": 0,
                "claims_supported": 0,
                "claims_contradicted": 0,
                "claims_unsupported": 0,
                "details": []
            }
        
        # Step 2: Verify each claim against source
        verified = self._verify_claims(claims, source_context)
        
        # Step 3: Calculate score
        supported = sum(1 for c in verified if c["status"] == "SUPPORTED")
        contradicted = sum(1 for c in verified if c["status"] == "CONTRADICTED")
        unsupported = sum(1 for c in verified if c["status"] == "UNSUPPORTED")
        
        total = len(verified)
        support_rate = (supported / total * 100) if total > 0 else 100
        
        return {
            "score": round(support_rate, 2),
            "verdict": "PASS" if support_rate >= 80 else "FAIL",
            "claims_total": total,
            "claims_supported": supported,
            "claims_contradicted": contradicted,
            "claims_unsupported": unsupported,
            "details": verified
        }
    
    def _extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from response using LLM."""
        
        prompt = f"""Extract all factual claims from this text. Return ONLY a JSON array of claim strings, nothing else.

Text: {response}

Format: ["claim 1", "claim 2", "claim 3"]

Output:"""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            content = completion.choices[0].message.content.strip()
            
            # Parse JSON
            claims = json.loads(content)
            return claims if isinstance(claims, list) else []
            
        except Exception as e:
            print(f"[WARN] Claim extraction failed: {e}")
            return []
    
    def _verify_claims(self, claims: List[str], source_context: str) -> List[Dict]:
        """Verify each claim against source context."""
        
        verified = []
        
        for claim in claims:
            status = self._verify_single_claim(claim, source_context)
            verified.append({
                "claim": claim,
                "status": status
            })
        
        return verified
    
    def _verify_single_claim(self, claim: str, source_context: str) -> str:
        """
        Verify single claim.
        
        Returns: "SUPPORTED" | "CONTRADICTED" | "UNSUPPORTED"
        """
        
        prompt = f"""Given the source context, is this claim supported, contradicted, or unsupported?

Source context:
{source_context}

Claim to verify:
{claim}

Answer with ONLY one word: SUPPORTED, CONTRADICTED, or UNSUPPORTED

- SUPPORTED: Claim is directly stated or clearly implied in source
- CONTRADICTED: Source explicitly contradicts the claim
- UNSUPPORTED: Claim is not mentioned in source (neither confirmed nor denied)

Answer:"""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            
            result = completion.choices[0].message.content.strip().upper()
            
            # Validate response
            if result in ["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]:
                return result
            else:
                return "UNSUPPORTED"  # Default fallback
                
        except Exception as e:
            print(f"[WARN] Claim verification failed: {e}")
            return "UNSUPPORTED"


# Singleton instance
_detector = None

def detect_hallucination(response: str, source_context: str) -> Dict:
    """
    Detect hallucinations (lazy-loaded singleton).
    
    Args:
        response: LLM response to check
        source_context: Ground truth context
        
    Returns:
        Hallucination detection results
    """
    global _detector
    if _detector is None:
        _detector = HallucinationDetector()
    
    return _detector.detect(response, source_context)