"""Tests for the tabular agent: SQL guardrails and the guarded text-to-SQL flow.

validate_sql tests are pure units; execute_guarded and run_tabular_agent use the
db_conn fixture (auto-skips when Postgres is down). No live LLM calls: the chat
model is always monkeypatched, both in common.llm and where agents.tabular imports it.
"""

from types import SimpleNamespace

import psycopg
import pytest

import common.llm
from agents import tabular
from agents.guardrails import GuardrailError, execute_guarded, validate_sql

# ------------------------------------------------------------------ validate_sql


def test_validate_sql_wraps_allowed_select() -> None:
    safe = validate_sql("SELECT a, b FROM civic_tbl_x WHERE a > 1;", {"civic_tbl_x"})
    assert safe == "SELECT * FROM (SELECT a, b FROM civic_tbl_x WHERE a > 1) AS guarded LIMIT 50"


def test_validate_sql_respects_row_limit() -> None:
    safe = validate_sql("SELECT * FROM civic_tbl_x", {"civic_tbl_x"}, row_limit=7)
    assert safe == "SELECT * FROM (SELECT * FROM civic_tbl_x) AS guarded LIMIT 7"


_INJECTIONS = [
    "SELECT * FROM civic_tbl_x; DROP TABLE sources; --",
    "SELECT * FROM civic_tbl_x UNION SELECT usename, passwd, 1 FROM pg_shadow",
    "UPDATE civic_tbl_x SET a = 1",
    "SELECT pg_sleep(10)",
    "SELECT 1 /* DROP */",
    "SELECT * FROM civic_tbl_other",  # not in the allowlist
    "SELECT 1",  # no civic_tbl_* reference at all
    "WITH x AS (DELETE FROM civic_tbl_x RETURNING 1) SELECT 1",
    "SELECT 1; SELECT 2",  # multi-statement without any forbidden keyword
]


@pytest.mark.parametrize("sql", _INJECTIONS)
def test_validate_sql_rejects_injections(sql: str) -> None:
    with pytest.raises(GuardrailError):
        validate_sql(sql, {"civic_tbl_x"})


# --------------------------------------------------------------- execute_guarded

_MESA = "civic_tbl_mesa_agenda_items_4474"
_MESA_COLUMNS = ["agenda_number", "agenda_sequence", "matter_file", "matter_type", "title"]


def _registry_rows(db_conn: psycopg.Connection) -> list[tuple]:
    rows = db_conn.execute(
        "SELECT table_name, description, columns_json FROM table_registry"
    ).fetchall()
    if not rows:
        pytest.skip("table_registry is empty (sample data not ingested)")
    return rows


def _require_mesa_table(db_conn: psycopg.Connection) -> None:
    names = {row[0] for row in _registry_rows(db_conn)}
    if _MESA not in names:
        pytest.skip(f"{_MESA} is not in table_registry")


def test_execute_guarded_columns_and_row_cap(db_conn: psycopg.Connection) -> None:
    _require_mesa_table(db_conn)
    columns, rows = execute_guarded(f"SELECT * FROM {_MESA}", {_MESA})
    assert columns == _MESA_COLUMNS
    assert len(rows) == 33  # full table fits under the default row_limit of 50
    _, capped = execute_guarded(f"SELECT * FROM {_MESA}", {_MESA}, row_limit=10)
    assert len(capped) == 10


# -------------------------------------------------------------- run_tabular_agent


class _FakeChatModel:
    """Stand-in for the chat model: canned reply or a raised exception, no network."""

    def __init__(self, reply: str = "", exc: Exception | None = None) -> None:
        self.reply = reply
        self.exc = exc
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(content=self.reply)


def _patch_llm(monkeypatch: pytest.MonkeyPatch, model: _FakeChatModel) -> None:
    """Patch both the source and the import site used by agents.tabular."""
    monkeypatch.setattr(common.llm, "get_chat_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(tabular, "get_chat_model", lambda *args, **kwargs: model)


_QUESTION = "How many agenda items are in the mesa agenda items table?"


def test_run_tabular_agent_with_valid_llm_sql(
    db_conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _registry_rows(db_conn)
    best_table = tabular._rank_tables(_QUESTION, rows)[0][0]
    model = _FakeChatModel(reply=f"```sql\nSELECT count(*) FROM {best_table}\n```")
    _patch_llm(monkeypatch, model)

    out = tabular.run_tabular_agent(_QUESTION)

    assert out["sql"] and all(isinstance(s, str) for s in out["sql"])
    (evidence,) = out["evidence"]
    assert evidence.kind == "table"
    assert "columns:" in evidence.text
    assert evidence.citation == f"[table: {best_table}]"
    assert evidence.meta["fallback"] is False
    assert model.prompts, "the fake LLM should have been consulted"


def test_run_tabular_agent_garbage_llm_falls_back(
    db_conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry_rows(db_conn)
    _patch_llm(monkeypatch, _FakeChatModel(reply="I cannot help"))

    out = tabular.run_tabular_agent(_QUESTION)

    (evidence,) = out["evidence"]
    assert evidence.meta["fallback"] is True
    assert "columns:" in evidence.text
    assert "row: " in evidence.text  # still returns rows via the fallback query
    assert out["sql"] and "LIMIT 10" in out["sql"][0]


def test_run_tabular_agent_llm_error_falls_back(
    db_conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry_rows(db_conn)
    _patch_llm(monkeypatch, _FakeChatModel(exc=RuntimeError("model down")))

    out = tabular.run_tabular_agent(_QUESTION)

    (evidence,) = out["evidence"]
    assert evidence.meta["fallback"] is True
    assert "row: " in evidence.text


def test_run_tabular_agent_injection_leaves_sources_intact(
    db_conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry_rows(db_conn)
    before = db_conn.execute("SELECT count(*) FROM sources").fetchone()
    assert before is not None
    _patch_llm(
        monkeypatch,
        _FakeChatModel(reply="SELECT * FROM civic_tbl_x; DROP TABLE sources; --"),
    )

    out = tabular.run_tabular_agent(_QUESTION)

    after = db_conn.execute("SELECT count(*) FROM sources").fetchone()
    assert after is not None
    assert before[0] == after[0]  # the injection never reached the database
    (evidence,) = out["evidence"]
    assert evidence.meta["fallback"] is True
    assert out["sql"] and "DROP" not in out["sql"][0]
