# judge/relevance_scorer.py
"""
Answer relevance scoring using semantic similarity.
Measures how well the response matches expected behavior.
"""

from typing import Dict
import numpy as np
from sentence_transformers import SentenceTransformer


class RelevanceScorer:
    """Score answer relevance using embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with sentence transformer model.
        
        Args:
            model_name: HuggingFace model for embeddings.
                       Default is lightweight 384-dim model.
        """
        self.model = SentenceTransformer(model_name)
    
    def score(self, response: str, expected_behavior: str) -> Dict:
        """
        Calculate relevance score.
        
        Args:
            response: LLM's actual response
            expected_behavior: Expected answer criteria
            
        Returns:
            Dict with score (0-100) and similarity details
        """
        # Generate embeddings
        response_emb = self.model.encode(response, convert_to_tensor=False)
        expected_emb = self.model.encode(expected_behavior, convert_to_tensor=False)
        
        # Cosine similarity
        cosine_sim = self._cosine_similarity(response_emb, expected_emb)
        
        # Scale to 0-100
        score = round(cosine_sim * 100, 2)
        
        return {
            "score": score,
            "cosine_similarity": round(cosine_sim, 4),
            "verdict": "PASS" if score >= 70 else "FAIL",
            "method": "semantic_similarity",
            "model": self.model_name
        }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return float(dot_product / (norm1 * norm2))
    
    @property
    def model_name(self) -> str:
        """Get model name safely — SentenceTransformer attr varies by version."""
        for attr in ("_model_card_vars", "model_card_data"):
            d = getattr(self.model, attr, {})
            if isinstance(d, dict) and d.get("model_name"):
                return d["model_name"]
        tok = getattr(self.model, "tokenizer", None)
        if tok and hasattr(tok, "name_or_path"):
            return tok.name_or_path
        return "all-MiniLM-L6-v2"


# Singleton instance
_scorer = None

def get_relevance_score(response: str, expected_behavior: str) -> Dict:
    """
    Get relevance score (lazy-loaded singleton).
    
    Args:
        response: LLM response text
        expected_behavior: Expected criteria
        
    Returns:
        Score dict with verdict
    """
    global _scorer
    if _scorer is None:
        _scorer = RelevanceScorer()
    
    return _scorer.score(response, expected_behavior)