## Setup
1. Install Postgres 15+, create DB `ragdb`.
2. Enable extensions in DB:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS pg_trgm;
   ```
3. Apply schema:
   ```bash
   psql $DATABASE_URL -f schema.sql
   ```
4. Create `.env` from `config.example.env` and fill keys.
5. Install deps:
   ```bash
   pip install -r requirements.txt
   ```

## Ingest JSONL
```bash
python ingest.py /path/to/your.jsonl
```

## Run API
```bash
uvicorn app:app --reload --port 8000
```

### Endpoints
- `POST /search` -> hybrid + rerank candidates
- `POST /answer` -> grounded answer with inline citations

## Test locally (offline)
```bash
export TEST_MODE=1
pytest -q
```

## Tuning Notes
- Adjust `VECTOR(1536)` to your embedding size.
- `lists` in `ivfflat` ~ sqrt(N / 10). Run `ANALYZE passages;` after index build.
- Use `ts_rank_cd` weights to boost headings. Add `pg_trgm` fuzzy filters for titles.
- Add caching of embeddings and search results in Redis for lower latency.
- Add versioning on `docs_raw` and soft-delete with `ON DELETE CASCADE` on `passages`.

# End of starter kit