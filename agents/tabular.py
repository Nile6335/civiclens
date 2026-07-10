"""Text-to-SQL tabular agent over the normalized civic_tbl_* tables.

The LLM only *proposes* SQL; every candidate (including the deterministic fallback)
runs through agents.guardrails — application validation, the SELECT-only civiclens_ro
role, a statement timeout, and a hard row cap. The agent never raises: any failure
degrades to the fallback query or, at worst, to empty evidence.
"""

import logging
import re
from collections.abc import Sequence

from agents.evidence import Evidence
from agents.guardrails import GuardrailError, execute_guarded
from common.db import get_connection
from common.llm import get_chat_model
from retrieval.search import SearchFilters

logger = logging.getLogger(__name__)

TOP_TABLES = 2
FALLBACK_LIMIT = 10
MAX_RENDER_ROWS = 15

# (table_name, description, columns_json) straight from table_registry.
RegistryRow = tuple[str, str, list[dict]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.+?)```", re.IGNORECASE | re.DOTALL)
_SQL_START_RE = re.compile(r"\b(?:select|with)\b", re.IGNORECASE)

_SQL_PROMPT = """You translate a question about city-council data into ONE PostgreSQL \
SELECT query.

Available tables:
{schema_block}

Rules:
- Reply with the SQL only — a single SELECT (or WITH) statement.
- No explanation, no semicolon, no comments.
- Use only the tables and columns listed above.
- "How many ..." means count rows: use count(*). Never use DISTINCT unless the
  question explicitly says distinct or unique.

Example:
Question: How many items were on the agenda?
SQL: SELECT count(*) FROM civic_tbl_example_agenda_items

Example:
Question: What is the title of agenda item 5?
SQL: SELECT title FROM civic_tbl_example_agenda_items WHERE agenda_number = '5'

Question: {question}
SQL:"""


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _rank_tables(question: str, registry_rows: Sequence[RegistryRow]) -> list[RegistryRow]:
    """Order registry rows by keyword overlap with the question (name tie-break)."""
    question_tokens = _tokens(question)

    def sort_key(row: RegistryRow) -> tuple[int, str]:
        table_name, description = row[0], row[1]
        overlap = len(question_tokens & (_tokens(table_name) | _tokens(description)))
        return (-overlap, table_name)

    return sorted(registry_rows, key=sort_key)


def _load_registry(filters: SearchFilters | None) -> list[RegistryRow]:
    """Load table_registry rows, joined against sources for city filtering."""
    sql = "SELECT tr.table_name, tr.description, tr.columns_json FROM table_registry tr"
    params: list[object] = []
    if filters is not None and filters.city:
        sql += " JOIN sources s ON s.id = tr.source_id WHERE s.city = %s"
        params.append(filters.city)
    sql += " ORDER BY tr.table_name"
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()  # type: ignore[arg-type]
    return [(row[0], row[1], row[2]) for row in rows]


def _schema_block(rows: Sequence[RegistryRow]) -> str:
    blocks = []
    for table_name, description, columns in rows:
        cols = ", ".join(f"{col['name']} ({col['sql_type']})" for col in columns)
        blocks.append(f"- {table_name}: {description}\n  columns: {cols}")
    return "\n".join(blocks)


def _extract_sql(reply: str) -> str | None:
    """Pull a SQL candidate out of an LLM reply: ```sql fence, else first SELECT/WITH."""
    text = reply.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    match = _SQL_START_RE.search(text)
    if not match:
        return None
    return text[match.start() :].strip()


def _propose_sql(question: str, ranked: Sequence[RegistryRow]) -> str | None:
    """Ask the LLM for one SELECT; None on empty/garbage output or LLM failure."""
    prompt = _SQL_PROMPT.format(schema_block=_schema_block(ranked), question=question)
    try:
        reply = str(get_chat_model().invoke(prompt).content)
    except Exception as exc:
        logger.warning("tabular agent LLM call failed: %s", exc)
        return None
    return _extract_sql(reply)


def _render(table: str, sql: str, columns: list[str], rows: list[tuple]) -> str:
    lines = [f"Query over {table}: {sql}", "columns: " + " | ".join(columns)]
    lines.extend("row: " + " | ".join(str(v) for v in row) for row in rows[:MAX_RENDER_ROWS])
    return "\n".join(lines)


def run_tabular_agent(question: str, filters: SearchFilters | None = None) -> dict:
    """Guarded text-to-SQL: {"evidence": [Evidence], "sql": [executed SQL]}. Never raises."""
    try:
        return _run(question, filters)
    except Exception:
        logger.exception("tabular agent failed")
        return {"evidence": [], "sql": []}


def _run(question: str, filters: SearchFilters | None) -> dict:
    registry = _load_registry(filters)
    if not registry:
        return {"evidence": [], "sql": []}
    ranked = _rank_tables(question, registry)[:TOP_TABLES]
    allowed_tables = {row[0] for row in ranked}
    best_table = ranked[0][0]

    fallback = False
    executed_sql = _propose_sql(question, ranked)
    columns: list[str] = []
    rows: list[tuple] = []
    if executed_sql is not None:
        try:
            columns, rows = execute_guarded(executed_sql, allowed_tables)
        except GuardrailError as exc:
            logger.info("tabular agent: candidate SQL rejected (%s); falling back", exc)
            executed_sql = None
    if executed_sql is None:
        fallback = True
        executed_sql = f"SELECT * FROM {best_table} LIMIT {FALLBACK_LIMIT}"
        try:
            columns, rows = execute_guarded(executed_sql, allowed_tables)
        except GuardrailError as exc:
            logger.warning("tabular agent: fallback query failed (%s)", exc)
            return {"evidence": [], "sql": []}

    evidence = Evidence.from_table(
        text=_render(best_table, executed_sql, columns, rows),
        table_name=best_table,
        sql=executed_sql,
        fallback=fallback,
    )
    return {"evidence": [evidence], "sql": [executed_sql]}
