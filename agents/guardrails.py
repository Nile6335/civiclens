"""Layered SQL guardrails for the tabular agent.

Defense in depth, in order:
1. Application validation (validate_sql): a single SELECT/WITH statement, no comments,
   no write/DDL/session keywords, no system catalogs, allowlisted civic_tbl_* tables
   only, wrapped in a hard LIMIT.
2. The SELECT-only civiclens_ro database role (infra/migrations/001_init.sql).
3. A statement timeout applied inside the transaction at execution time.
"""

import logging
import re

import psycopg

from common.db import get_connection

logger = logging.getLogger(__name__)

STATEMENT_TIMEOUT = "3000ms"

_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "grant",
    "revoke",
    "truncate",
    "copy",
    "vacuum",
    "call",
    "execute",
    "prepare",
    "deallocate",
    "listen",
    "notify",
    "refresh",
    "reindex",
    "cluster",
    "lock",
    "merge",
    "set",
    "reset",
    "security",
    "pg_sleep",
    "pg_read_file",
    "pg_terminate_backend",
    "dblink",
    "lo_import",
)

_FORBIDDEN_RE = re.compile(r"\b(?:" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
# pg_catalog and every other pg_* identifier are covered by the pg_ prefix rule.
_SYSTEM_RE = re.compile(r"\b(?:information_schema|pg_[a-z0-9_]+)\b", re.IGNORECASE)
_SELECT_START_RE = re.compile(r"^(?:select|with)\b", re.IGNORECASE)
_CIVIC_TBL_RE = re.compile(r"\bcivic_tbl_[a-z0-9_]+\b", re.IGNORECASE)


class GuardrailError(Exception):
    """A candidate SQL query failed validation or guarded execution."""


def validate_sql(sql: str, allowed_tables: set[str], row_limit: int = 50) -> str:
    """Validate an untrusted SELECT and return it wrapped in a hard row cap.

    Raises GuardrailError on anything that is not a single read-only SELECT/WITH
    over allowlisted civic_tbl_* tables.
    """
    query = sql.strip()
    if query.endswith(";"):
        query = query[:-1].rstrip()
    if ";" in query:
        raise GuardrailError("multi-statement SQL is not allowed")
    if "--" in query or "/*" in query:
        raise GuardrailError("SQL comments are not allowed")
    if not _SELECT_START_RE.match(query):
        raise GuardrailError("only SELECT/WITH queries are allowed")
    forbidden = _FORBIDDEN_RE.search(query)
    if forbidden:
        raise GuardrailError(f"forbidden keyword: {forbidden.group(0).lower()}")
    system = _SYSTEM_RE.search(query)
    if system:
        raise GuardrailError(f"system catalog access is not allowed: {system.group(0).lower()}")
    tables = {match.lower() for match in _CIVIC_TBL_RE.findall(query)}
    if not tables:
        raise GuardrailError("query must reference at least one civic_tbl_* table")
    if not tables <= allowed_tables:
        unknown = ", ".join(sorted(tables - allowed_tables))
        raise GuardrailError(f"table(s) not in the allowlist: {unknown}")
    return f"SELECT * FROM ({query}) AS guarded LIMIT {int(row_limit)}"


def execute_guarded(
    sql: str, allowed_tables: set[str], row_limit: int = 50
) -> tuple[list[str], list[tuple]]:
    """validate_sql, then run on the read-only role with a statement timeout.

    Returns (column names, rows). Any psycopg error is re-raised as a GuardrailError
    with a short message (never a full traceback).
    """
    safe_sql = validate_sql(sql, allowed_tables, row_limit)
    try:
        with get_connection(readonly=True) as conn, conn.transaction():
            conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
            cursor = conn.execute(safe_sql)  # type: ignore[arg-type]
            columns = [col.name for col in cursor.description or []]
            rows = cursor.fetchall()
    except psycopg.Error as exc:
        lines = str(exc).strip().splitlines()
        detail = lines[0] if lines else type(exc).__name__
        logger.warning("guarded query failed: %s", detail)
        raise GuardrailError(f"query execution failed: {detail}") from exc
    return columns, rows
