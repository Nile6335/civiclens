"""Database helpers: connections, migrations, and the embedding-dimension guard."""

import contextlib
import logging
import re
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from common.settings import get_settings

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "infra" / "migrations"


def get_connection(autocommit: bool = False, readonly: bool = False) -> psycopg.Connection:
    """Open a psycopg connection with pgvector type adaptation registered."""
    settings = get_settings()
    dsn = settings.database_url_ro if readonly else settings.database_url
    conn = psycopg.connect(dsn, autocommit=autocommit)
    # vector extension may not exist pre-migration; callers that need it fail loudly later
    with contextlib.suppress(psycopg.ProgrammingError):
        register_vector(conn)
    return conn


def run_migrations() -> list[str]:
    """Apply infra/migrations/*.sql in filename order, tracking applied ones.

    SQL files may contain the {EMBEDDING_DIM} placeholder, substituted from settings
    (a pgvector column's dimension is fixed at CREATE time).
    """
    settings = get_settings()
    applied: list[str] = []
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  filename TEXT PRIMARY KEY,"
            "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )
        done = {row[0] for row in conn.execute("SELECT filename FROM schema_migrations").fetchall()}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            sql = path.read_text().replace("{EMBEDDING_DIM}", str(settings.embedding_dim))
            logger.info("applying migration %s", path.name)
            conn.execute(sql)  # type: ignore[arg-type]
            conn.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
            applied.append(path.name)
        _record_embedding_config(conn)
    return applied


def _record_embedding_config(conn: psycopg.Connection) -> None:
    """Persist the embedding model/dim the schema was created with, and guard mismatches."""
    settings = get_settings()
    rows = dict(
        conn.execute(
            "SELECT key, value FROM civiclens_meta"
            " WHERE key IN ('embedding_model', 'embedding_dim')"
        ).fetchall()
    )
    if rows:
        stored_dim = int(rows.get("embedding_dim", settings.embedding_dim))
        if stored_dim != settings.embedding_dim:
            raise RuntimeError(
                f"Schema was created with EMBEDDING_DIM={stored_dim} "
                f"({rows.get('embedding_model')}), but settings now say "
                f"EMBEDDING_DIM={settings.embedding_dim} ({settings.embedding_model}). "
                "Reset the database (make db-reset) or restore the old settings."
            )
        return
    conn.execute(
        "INSERT INTO civiclens_meta (key, value) VALUES "
        "('embedding_model', %s), ('embedding_dim', %s) "
        "ON CONFLICT (key) DO NOTHING",
        (settings.embedding_model, str(settings.embedding_dim)),
    )


_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def quote_ident(name: str) -> str:
    """Validate and return a safe SQL identifier (used for dynamic civic_tbl_* tables)."""
    if not _IDENT_RE.match(name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return f'"{name}"'


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    applied = run_migrations()
    print(f"migrations applied: {applied or 'none (up to date)'}")


if __name__ == "__main__":
    main()
