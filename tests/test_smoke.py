"""Phase 0 acceptance: trivial smoke test + pgvector confirmed via a real query."""

from common.settings import get_settings


def test_smoke() -> None:
    assert 1 + 1 == 2


def test_settings_load() -> None:
    settings = get_settings()
    assert settings.llm_backend in ("ollama", "anthropic")
    assert settings.embedding_dim > 0


def test_pgvector_extension(db_conn) -> None:
    row = db_conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert row is not None, "pgvector extension is not installed"


def test_pgvector_roundtrip(db_conn) -> None:
    """A real vector query: cosine distance between two literal vectors."""
    row = db_conn.execute("SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector").fetchone()
    assert row is not None
    assert abs(row[0] - 1.0) < 1e-6  # orthogonal vectors → cosine distance 1


def test_core_tables_exist(db_conn) -> None:
    rows = db_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert {"sources", "chunks", "table_registry", "civiclens_meta"} <= names
