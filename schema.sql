-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;         -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;        -- trigram for fuzzy titles

-- Raw JSON store (source of truth)
CREATE TABLE IF NOT EXISTS docs_raw (
  id           BIGSERIAL PRIMARY KEY,
  filename     TEXT,
  title        TEXT,
  year         INT,
  category     TEXT,
  data         JSONB NOT NULL,
  created_at   TIMESTAMPTZ DEFAULT now(),
  updated_at   TIMESTAMPTZ DEFAULT now()
);

-- Retrieval passages (chunked + embedded)
CREATE TABLE IF NOT EXISTS passages (
  id           BIGSERIAL PRIMARY KEY,
  doc_id       BIGINT REFERENCES docs_raw(id) ON DELETE CASCADE,
  section_no   TEXT,
  heading      TEXT,
  text         TEXT NOT NULL,
  embedding    VECTOR(768),                 -- adjust to your embedding dim
  -- denormalized filters
  year         INT,
  category     TEXT,
  -- housekeeping
  token_count  INT,
  checksum     TEXT,                         -- to avoid duplicate chunks
  created_at   TIMESTAMPTZ DEFAULT now(),
  updated_at   TIMESTAMPTZ DEFAULT now()
);

-- Lexical full-text index (heading weighted higher than text)
CREATE INDEX IF NOT EXISTS passages_tsv_idx ON passages
USING GIN (
  to_tsvector('english', coalesce(heading,'') || ' ' || text)
);

-- Trigram on headings for fuzzy queries
CREATE INDEX IF NOT EXISTS passages_heading_trgm_idx ON passages
USING GIN (heading gin_trgm_ops);

-- Vector ANN index (IVFFLAT). Build after inserting some rows.
-- Tune lists per data size; ANALYZE after build.
CREATE INDEX IF NOT EXISTS passages_vec_idx ON passages
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);

-- Common filter indexes
CREATE INDEX IF NOT EXISTS passages_year_idx ON passages(year);
CREATE INDEX IF NOT EXISTS passages_category_idx ON passages(category);
CREATE INDEX IF NOT EXISTS docs_raw_title_trgm_idx ON docs_raw USING GIN (title gin_trgm_ops);