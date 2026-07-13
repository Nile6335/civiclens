-- PII redaction quarantine (Phase 8): originals of redacted transcript chunks are
-- retained here, and ONLY here. The tabular agent's read-only role must never see
-- them; access requires the owner connection.

CREATE TABLE IF NOT EXISTS pii_quarantine (
    id SERIAL PRIMARY KEY,
    city TEXT NOT NULL,
    source_type TEXT NOT NULL,
    meeting_id TEXT,
    chunk_index INT NOT NULL,
    original_text TEXT NOT NULL,
    redacted_text TEXT NOT NULL,
    spans JSONB NOT NULL,  -- [{"type": "phone|email|address|person", "start": int, "end": int, "text": str}]
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (city, source_type, meeting_id, chunk_index)
);

REVOKE ALL ON pii_quarantine FROM civiclens_ro;
