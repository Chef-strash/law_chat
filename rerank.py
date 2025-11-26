import os
import math
from typing import List, Dict, Optional
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# --- PRODUCTION CONFIGURATION ---
# Switch to the V2-M3 model (Multilingual, Lightweight, High Accuracy)
RERANK_MODEL = os.getenv('RERANK_MODEL', 'BAAI/bge-reranker-v2-m3')
TEST_MODE = os.getenv('TEST_MODE', '0') == '1'

_reranker = None

def get_reranker():
    """
    Lazy-load CrossEncoder reranker.
    """
    global _reranker
    if TEST_MODE:
        return None
    
    if _reranker is None:
        print(f"⚖️  Loading Reranker: {RERANK_MODEL}...")
        from sentence_transformers import CrossEncoder
        # trust_remote_code=True is often needed for newer BGE models
        _reranker = CrossEncoder(RERANK_MODEL, trust_remote_code=True)
    return _reranker

def _local_score(q: str, t: str) -> float:
    """
    Simple Jaccard over tokens for TEST_MODE / CI.
    """
    qs = set(q.lower().split())
    ts = set(t.lower().split())
    if not qs or not ts: return 0.0
    inter = len(qs & ts)
    union = len(qs | ts)
    return inter / union

def _sigmoid(x: float) -> float:
    """
    Converts raw logits to a 0-1 probability score.
    Critical for consistent Web Search fallback thresholds.
    """
    return 1 / (1 + math.exp(-x))

def _build_text_for_rerank(c: Dict) -> str:
    """
    Constructs the text pair for the Cross Encoder.
    
    STRATEGY:
    We prioritize 'search_hit' (the Child Chunk) if it exists.
    This is the specific clause/paragraph that matched the query.
    We include Title and Heading for context.
    """
    title = c.get("title") or ""
    heading = c.get("heading") or ""
    
    # Use the specific matching chunk (Child) for scoring if available.
    # If not (legacy data), fall back to the main text.
    # Note: 'text' in candidate dict is now the PARENT context (per search.py updates),
    # so we prefer 'search_hit' for relevance scoring.
    body = c.get("search_hit") if c.get("search_hit") else c.get("text")
    body = body or ""
    
    # Format: "Title > Heading: Body"
    # This hierarchical format helps BGE models understand structure.
    combined = f"{title} > {heading}: {body}".strip()
    return combined

def rerank(
    query: str,
    candidates: List[Dict],
    top_n: int = 10,
    threshold: Optional[float] = None
) -> List[Dict]:
    """
    Reranks candidates using Cross-Encoder.
    Returns the top_n results with normalized scores (0-1).
    """
    if not candidates:
        return []

    rr = get_reranker()

    if rr is None:  # TEST_MODE
        for c in candidates:
            text_for_score = _build_text_for_rerank(c)
            c['rerank'] = _local_score(query, text_for_score)
    else:
        # 1. Prepare Pairs
        pairs = [(query, _build_text_for_rerank(c)) for c in candidates]
        
        # 2. Predict (returns raw logits)
        raw_scores = rr.predict(pairs)
        
        # Handle single result vs list of results
        if not isinstance(raw_scores, (list, np.ndarray)):
            raw_scores = [raw_scores]
        
        scores_list = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores

        # 3. Assign & Normalize Scores
        for c, s in zip(candidates, scores_list):
            # Apply Sigmoid to get 0-1 range
            normalized_score = _sigmoid(float(s))
            c['rerank'] = normalized_score
            c['raw_score'] = float(s) # Keep raw score for debugging if needed

    # 4. Sort by Rerank Score
    candidates.sort(key=lambda x: x['rerank'], reverse=True)
    
    # 5. Filter & Slice
    out = candidates
    if threshold is not None:
        out = [c for c in out if c['rerank'] >= threshold]

    return out[:top_n]

