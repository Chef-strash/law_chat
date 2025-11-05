import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
RERANK_MODEL = os.getenv('RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
TEST_MODE = os.getenv('TEST_MODE', '0') == '1'

_reranker = None

def get_reranker():
    global _reranker
    if TEST_MODE:
        return None
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL, trust_remote_code=True)
    return _reranker

# Input: query str, candidates = list[{id, text, title, heading, ...}]
# Output: top_n with scores >= threshold if provided

def _local_score(q: str, t: str) -> float:
    # simple Jaccard over tokens for tests
    qs = set(q.lower().split())
    ts = set(t.lower().split())
    if not qs or not ts:
        return 0.0
    inter = len(qs & ts)
    union = len(qs | ts)
    return inter / union


def rerank(query: str, candidates: List[Dict], top_n=10, threshold: float | None = None) -> List[Dict]:
    rr = get_reranker()
    if rr is None:  # TEST_MODE
        for c in candidates:
            c['rerank'] = _local_score(query, c['text'])
    else:
        pairs = [(query, c['text']) for c in candidates]
        scores = rr.predict(pairs).tolist()
        for c, s in zip(candidates, scores):
            c['rerank'] = float(s)
    candidates.sort(key=lambda x: x['rerank'], reverse=True)
    out = candidates[:top_n]
    if threshold is not None:
        out = [c for c in out if c['rerank'] >= threshold]
    return out