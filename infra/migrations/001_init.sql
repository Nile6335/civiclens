-- Core schema. {EMBEDDING_DIM} is substituted by the migration runner from settings,
-- because a pgvector column's dimension is fixed at creation time.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS civiclens_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id SERIAL PRIMARY KEY,
    city TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('transcript', 'pdf', 'table')),
    meeting_id TEXT,
    title TEXT NOT NULL,
    url TEXT,
    meeting_date DATE,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (city, source_type, meeting_id, title)
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    source_id INT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    t_start REAL,
    t_end REAL,
    page_no INT,
    topic TEXT,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding vector({EMBEDDING_DIM}),
    UNIQUE (source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- Registry of normalized tabular sources (budget/vote tables) for the tabular agent.
-- Actual data lives in dynamically created tables named civic_tbl_*.
CREATE TABLE IF NOT EXISTS table_registry (
    id SERIAL PRIMARY KEY,
    source_id INT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    table_name TEXT NOT NULL UNIQUE CHECK (table_name ~ '^civic_tbl_[a-z0-9_]+$'),
    description TEXT NOT NULL,
    columns_json JSONB NOT NULL
);

-- Read-only role used by the tabular agent as a hard guardrail (SELECT-only at the
-- database level, on top of application-level SQL validation).
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'civiclens_ro') THEN
        CREATE ROLE civiclens_ro LOGIN PASSWORD 'civiclens_ro';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE civiclens TO civiclens_ro;
GRANT USAGE ON SCHEMA public TO civiclens_ro;
GRANT SELECT ON table_registry TO civiclens_ro;
-- SELECT on civic_tbl_* tables is granted at creation time by the ingestion pipeline.
