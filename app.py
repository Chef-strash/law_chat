import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
from search import hybrid_search
from rerank import rerank

TEST_MODE = os.getenv('TEST_MODE', '0') == '1'

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    pre_k: int = 200
    mmr_k: int = 20
    top_n: int = 8
    threshold: Optional[float] = None

@app.post('/search')
async def api_search(req: SearchRequest):
    cands = hybrid_search(req.query, req.filters, pre_k=req.pre_k, mmr_k=req.mmr_k)
    ranked = rerank(req.query, cands, top_n=req.top_n, threshold=req.threshold)
    return {'results': ranked}

# Simple grounded answer using top passages
PROMPT = (
    "You are a factual assistant. Answer the user query ONLY using the provided passages.\n"
    "Cite sources inline as [title §section_no] or [title]. If unsure, say you cannot find it.\n\n"
    "Question: {q}\n\nPassages:\n{ctx}\n"
)

class AnswerRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    top_n: int = 8

@app.post('/answer')
async def api_answer(req: AnswerRequest):
    cands = hybrid_search(req.query, req.filters, pre_k=200, mmr_k=40)
    ranked = rerank(req.query, cands, top_n=req.top_n)
    
    # Build context with minimal leakage, include citations
    ctx_lines = []
    for i, r in enumerate(ranked, start=1):
        cite = r.get('title') or 'Source'
        sec = r.get('heading') or ''
        ctx_lines.append(f"[{i}] ({cite}) {sec}\n{r['text'][:1200]}")

    prompt = PROMPT.format(q=req.query, ctx='\n\n'.join(ctx_lines))

    from openai import OpenAI
    client = OpenAI()
    comp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0
    )
    answer = comp.choices[0].message.content
    return {'answer': answer, 'citations': ranked}