⚖️ Legal RAG System: Indian Constitutional Law

A production-grade Retrieval-Augmented Generation (RAG) system engineered for high-precision legal research. This system is designed to navigate the complexities of Indian Constitutional Law, utilizing advanced retrieval strategies, semantic reranking, and an intelligent web fallback mechanism.

🌟 Key Features

Hybrid Search Engine: Combines HNSW Vector Search (Semantic) with PostgreSQL TSVECTOR (Lexical/Keyword) to capture both conceptual similarity and specific legal terminology.

Parent-Child Retrieval: Indexes small "Child" chunks for precise search hits but retrieves the full "Parent" section/article for the LLM, ensuring complete legal context.

State-of-the-Art Models:

Embeddings: BAAI/bge-m3 (1024 dimensions) for multilingual and long-context understanding.

Reranking: BAAI/bge-reranker-v2-m3 for assessing relevance with high accuracy.

Intelligent Web Fallback: Automatically switches to a live web search (via Tavily) if the internal database confidence drops below a strict threshold.

Source Citations: Distinguished citations for Database sources (Verified) vs. Web sources.

🏗️ Architecture Workflow

graph TD
    A[User Query] --> B[Hybrid Search (BM25 + HNSW)]
    B --> C[Top 200 Candidates]
    C --> D[Cross-Encoder Reranking]
    D --> E{Confidence Score > 0.45?}
    
    E -- Yes --> F[Retrieve Parent Context from DB]
    E -- No --> G[Trigger Tavily Web Search]
    
    F --> H[Synthesize Prompt]
    G --> H
    
    H --> I[Google Gemini 1.5 Pro]
    I --> J[Final Legal Answer with Citations]


📂 Project Structure

legal-rag/
├── app.py                  # Main FastAPI application (The Brain)
├── search.py               # Hybrid search logic & DB retrieval
├── rerank.py               # Cross-encoder reranking logic
├── web_search.py           # Tavily integration for web fallback
├── ingest.py               # Data ingestion script (ETL pipeline)
├── models.py               # SQLAlchemy Database Models
├── schema.sql              # Raw SQL schema (optional reference)
├── requirements.txt        # Python dependencies
├── .env                    # Configuration secrets
└── data/                   # STORE YOUR SCRAPED FILES HERE
    └── constitutional_law.jsonl


🚀 Setup & Installation

1. Prerequisites

Python 3.10+

PostgreSQL 14+

pgvector extension installed in Postgres.

2. Install Dependencies

pip install -r requirements.txt


3. Database Setup

Make sure your PostgreSQL service is running. Then create the database:

# Enter Postgres shell
psql postgres

# Create DB
CREATE DATABASE rag_db;

# Connect and enable Vector extension
\c rag_db
CREATE EXTENSION vector;


4. Configuration

Create a .env file in the root directory:

DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
GOOGLE_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key

# Production Models
EMBED_MODEL=BAAI/bge-m3
RERANK_MODEL=BAAI/bge-reranker-v2-m3
LLM_MODEL=gemini-1.5-pro


💾 Data Ingestion

This is where you load your scraped Indian Kanoon data.

Step 1: Format Your Data

Save your scraped data into a .jsonl (JSON Lines) file inside the data/ folder. Each line must be a valid JSON object.

Required Format:

{"title": "Constitution of India", "heading": "Article 21", "text": "Protection of life and personal liberty...", "year": 1950, "category": "Fundamental Rights"}


text: This should be the full text of the Article or Section. The system will automatically split this into smaller "child" chunks for searching while preserving this text as the "parent" context.

Step 2: Run Ingestion

This script will initialize the database tables, chunk the text, generate embeddings (1024-dim), and store them in Postgres.

python ingest.py data/constitutional_law.jsonl


Depending on the size of your dataset and GPU availability, this may take some time.

⚡ Running the Application

Start the FastAPI server:

uvicorn app:app --reload --host 0.0.0.0 --port 8000


API URL: http://localhost:8000

Interactive Docs: http://localhost:8000/docs

🔌 API Usage

1. Generate Legal Answer (RAG)

This endpoint handles the full logic: Search -> Rerank -> Decision -> Answer.

Request: POST /answer

{
  "query": "Can a person be prosecuted twice for the same offence?",
  "filters": {
    "category": "Fundamental Rights"
  },
  "top_n": 5
}


Response:

{
  "answer": "No, per Article 20(2) of the Constitution, no person shall be prosecuted and punished for the same offence more than once...",
  "sources": [
    {
      "id": 105,
      "title": "Constitution of India",
      "heading": "Article 20",
      "text": "Protection in respect of conviction for offences...",
      "type": "db",
      "score": 0.89
    }
  ],
  "source_type": "database",
  "confidence": 0.89
}


2. Raw Search (Debug)

Use this to check what documents are being retrieved before the LLM sees them.

Request: POST /search

{
  "query": "basic structure doctrine",
  "top_n": 10
}