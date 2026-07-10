"""Single writer for ingestion output: sources, chunks, and normalized tables."""

import json
import logging

import psycopg

from common.db import quote_ident
from ingestion.models import ChunkRecord, NormalizedTable, SourceRecord

logger = logging.getLogger(__name__)

_SQL_TYPES = {"text": "TEXT", "numeric": "NUMERIC", "integer": "INTEGER", "date": "DATE"}


def upsert_source(conn: psycopg.Connection, source: SourceRecord) -> int:
    """Insert or refresh a source row; returns its id. Idempotent on the natural key."""
    row = conn.execute(
        """
        INSERT INTO sources (city, source_type, meeting_id, title, url, meeting_date, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (city, source_type, meeting_id, title)
        DO UPDATE SET url = EXCLUDED.url,
                      meeting_date = EXCLUDED.meeting_date,
                      metadata = EXCLUDED.metadata
        RETURNING id
        """,
        (
            source.city,
            source.source_type,
            source.meeting_id,
            source.title,
            source.url,
            source.meeting_date,
            json.dumps(source.metadata),
        ),
    ).fetchone()
    assert row is not None
    return row[0]


def replace_chunks(conn: psycopg.Connection, source_id: int, chunks: list[ChunkRecord]) -> int:
    """Replace all chunks for a source (re-ingestion is idempotent). Returns count."""
    conn.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (source_id, chunk_index, text, t_start, t_end, page_no, topic)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (source_id, c.chunk_index, c.text, c.t_start, c.t_end, c.page_no, c.topic)
                for c in chunks
            ],
        )
    return len(chunks)


def create_normalized_table(
    conn: psycopg.Connection, source_id: int, table: NormalizedTable
) -> str:
    """Create/replace a civic_tbl_* table with its data and register it. Returns table name."""
    table_name = f"civic_tbl_{table.slug}"
    quoted = quote_ident(table_name)
    cols_sql = ", ".join(
        f"{quote_ident(col.name)} {_SQL_TYPES[col.sql_type]}" for col in table.columns
    )
    conn.execute(f"DROP TABLE IF EXISTS {quoted}")  # idempotent re-ingestion
    conn.execute(f"CREATE TABLE {quoted} ({cols_sql})")
    placeholders = ", ".join(["%s"] * len(table.columns))
    with conn.cursor() as cur:
        cur.executemany(f"INSERT INTO {quoted} VALUES ({placeholders})", table.rows)
    conn.execute(f"GRANT SELECT ON {quoted} TO civiclens_ro")
    conn.execute(
        """
        INSERT INTO table_registry (source_id, table_name, description, columns_json)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (table_name)
        DO UPDATE SET source_id = EXCLUDED.source_id,
                      description = EXCLUDED.description,
                      columns_json = EXCLUDED.columns_json
        """,
        (
            source_id,
            table_name,
            table.description,
            json.dumps([{"name": c.name, "sql_type": c.sql_type} for c in table.columns]),
        ),
    )
    logger.info("normalized table %s created with %d rows", table_name, len(table.rows))
    return table_name


def corpus_stats(conn: psycopg.Connection) -> dict:
    """Row counts for CLI logging and ingest summaries."""
    stats: dict = {}
    for key, sql in {
        "sources": "SELECT count(*) FROM sources",
        "chunks": "SELECT count(*) FROM chunks",
        "tables": "SELECT count(*) FROM table_registry",
        "chunks_by_type": (
            "SELECT s.source_type, count(*) FROM chunks c"
            " JOIN sources s ON s.id = c.source_id GROUP BY 1 ORDER BY 1"
        ),
    }.items():
        rows = conn.execute(sql).fetchall()
        stats[key] = dict(rows) if key == "chunks_by_type" else rows[0][0]
    return stats
