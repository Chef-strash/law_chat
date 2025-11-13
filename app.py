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

_llm = None

def get_llm():

    global _llm
    if _llm is None:
        from langchain.chat_models import init_chat_model
        
        # Check if API key is set
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY not found in environment. "
            )
                
        _llm = init_chat_model(
            model_name="gemini-2.5-flash",
            model_provider="google_genai",
            temperature=0.2
        )
        print(f"✓ Gemini LLM initialized")
    
    return _llm

# RAG prompt template
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on provided context.

Instructions:
- Answer the question using ONLY the information from the passages below
- Cite sources using [Source N] format where N is the source number
- If the passages don't contain the answer, say "I cannot find this information in the provided context"
- Be concise and factual

Question: {question}

Context:
{context}

Answer:"""

class AnswerRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    top_n: int = 8

@app.post('/answer')
async def api_answer(req: AnswerRequest):
    """Generate answer using free Google Gemini LLM."""
    
    # Get relevant passages
    cands = hybrid_search(req.query, req.filters, pre_k=200, mmr_k=40)
    ranked = rerank(req.query, cands, top_n=req.top_n)
    
    if not ranked:
        return {
            'answer': 'No relevant information found.',
            'citations': []
        }
    
    # Build context from top passages
    ctx_lines = []
    for i, r in enumerate(ranked, start=1):
        title = r.get('title') or 'Source'
        heading = r.get('heading') or ''
        text = r['text'][:1000]  # Limit per passage
        
        if heading:
            ctx_lines.append(f"[Source {i}] {title} - {heading}\n{text}")
        else:
            ctx_lines.append(f"[Source {i}] {title}\n{text}")
    
    context = '\n\n'.join(ctx_lines)
    
    # Format prompt
    prompt = PROMPT_TEMPLATE.format(
        question=req.query,
        context=context
    )
    
    # Generate answer with Gemini
    if TEST_MODE:
        answer = f"[TEST MODE] Mock answer for: {req.query}"
    else:
        try:
            llm = get_llm()
            response = llm.invoke(prompt)
            
            # Extract text from response
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
        except Exception as e:
            return {
                'error': f"LLM generation failed: {str(e)}",
                'citations': ranked
            }
    
    return {
        'answer': answer,
        'citations': ranked,
        'model': os.getenv('LLM_MODEL', 'gemini-2.0-flash-exp')
    }