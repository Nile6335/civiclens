"""Tests for ingestion.tables: normalization units + the read-only role guardrail."""

from datetime import date
from pathlib import Path

import psycopg
import pytest

from common.db import get_connection
from ingestion import store
from ingestion.models import RawTable, SourceRecord
from ingestion.tables import (
    coerce,
    dedupe_idents,
    infer_column_type,
    load_csv_table,
    normalize_raw_table,
    snake_case_ident,
)


def test_snake_case_ident_basic() -> None:
    assert snake_case_ident("FY2026 Budget ($M)") == "fy2026_budget_m"
    assert snake_case_ident("  Total % Change  ") == "total_change"


def test_snake_case_ident_digit_and_empty_fallbacks() -> None:
    assert snake_case_ident("2026 Total") == "col_2026_total"
    assert snake_case_ident("") == "col"
    assert snake_case_ident("$%^") == "col"
    assert snake_case_ident("", fallback="field") == "field"


def test_snake_case_ident_truncates() -> None:
    assert len(snake_case_ident("x" * 100)) == 58


def test_dedupe_idents() -> None:
    assert dedupe_idents(["amount", "amount", "amount"]) == ["amount", "amount_2", "amount_3"]
    assert dedupe_idents(["a", "a_2", "a"]) == ["a", "a_2", "a_3"]
    assert dedupe_idents(["x", "y"]) == ["x", "y"]


def test_infer_integer_with_commas() -> None:
    assert infer_column_type(["1,204", "890", "-12", ""]) == "integer"


def test_infer_numeric_money_percent_parens() -> None:
    assert infer_column_type(["$412.50", "(1,234.56)", "3%"]) == "numeric"
    assert infer_column_type(["1,204", "3.5"]) == "numeric"


def test_infer_date_iso_and_us() -> None:
    assert infer_column_type(["2026-01-31", "1/31/2026"]) == "date"


def test_infer_text_fallbacks() -> None:
    assert infer_column_type(["Police", "410"]) == "text"
    assert infer_column_type(["2026-01-31", "not a date"]) == "text"
    assert infer_column_type([]) == "text"
    assert infer_column_type(["", "  "]) == "text"


def test_coerce_values() -> None:
    assert coerce("1,204", "integer") == 1204
    assert coerce("(1,234.56)", "numeric") == pytest.approx(-1234.56)
    assert coerce("$412.50", "numeric") == pytest.approx(412.5)
    assert coerce("3%", "numeric") == pytest.approx(3.0)
    assert coerce("1/31/2026", "date") == date(2026, 1, 31)
    assert coerce("2026-01-31", "date") == date(2026, 1, 31)
    assert coerce("  Police ", "text") == "Police"
    assert coerce("", "integer") is None
    assert coerce("   ", "date") is None
    assert coerce(None, "numeric") is None


def test_coerce_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        coerce("not a number", "integer")
    with pytest.raises(ValueError):
        coerce("410", "boolean")


def test_normalize_ragged_rows() -> None:
    raw = RawTable(
        header=["Name", "Amount"],
        rows=[
            ["Police", "410", "extra"],  # one cell too many -> trimmed
            ["Fire"],  # one cell short -> padded
            ["", ""],  # fully empty -> dropped
            ["Parks", "12"],
        ],
    )
    table = normalize_raw_table(raw, "t", "test")
    assert [c.name for c in table.columns] == ["name", "amount"]
    assert [c.sql_type for c in table.columns] == ["text", "integer"]
    assert table.rows == [["Police", 410], ["Fire", None], ["Parks", 12]]


def test_normalize_dedupes_headers() -> None:
    raw = RawTable(header=["Total", "Total", ""], rows=[["1", "2", "x"]])
    table = normalize_raw_table(raw, "t", "d")
    assert [c.name for c in table.columns] == ["total", "total_2", "col"]


def test_normalize_rejects_bad_slug() -> None:
    raw = RawTable(header=["A"], rows=[["1"]])
    for bad in ("Bad", "has space", "semi;colon", ""):
        with pytest.raises(ValueError):
            normalize_raw_table(raw, bad, "d")


def test_load_csv_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "budget.csv"
    csv_path.write_text(
        'Department,"FY2026 Budget ($M)",FTEs\nPolice,"$412.50","1,204"\nFire,"(3.20)",890\n',
        encoding="utf-8",
    )
    table = load_csv_table(csv_path, "csv_budget", "csv test")
    assert [c.name for c in table.columns] == ["department", "fy2026_budget_m", "ftes"]
    assert [c.sql_type for c in table.columns] == ["text", "numeric", "integer"]
    assert table.rows[0] == ["Police", pytest.approx(412.5), 1204]
    assert table.rows[1][1] == pytest.approx(-3.2)


_SLUG = "pytest_tables_budget"
_TABLE_NAME = f"civic_tbl_{_SLUG}"
_CITY = "pytestville"
_TITLE = "pytest budget table"


def _cleanup(conn: psycopg.Connection) -> None:
    conn.rollback()
    conn.execute(f"DROP TABLE IF EXISTS {_TABLE_NAME}")
    # registry rows cascade-delete with the source
    conn.execute("DELETE FROM sources WHERE city = %s AND title = %s", (_CITY, _TITLE))
    conn.commit()


def test_create_normalized_table_roundtrip(db_conn: psycopg.Connection) -> None:
    raw = RawTable(
        header=["Department", "FY2026 Budget ($M)", "FTEs", "Adopted"],
        rows=[
            ["Police", "$412.50", "1,204", "2026-06-30"],
            ["Fire", "(3.20)", "890", "6/30/2026"],
            ["", "", "", ""],
            ["Parks", "12.0", "310", ""],
        ],
    )
    table = normalize_raw_table(raw, _SLUG, "FY2026 budget by department (pytest fixture)")
    source = SourceRecord(city=_CITY, source_type="table", title=_TITLE)
    try:
        source_id = store.upsert_source(db_conn, source)
        name = store.create_normalized_table(db_conn, source_id, table)
        db_conn.commit()
        assert name == _TABLE_NAME

        ro = get_connection(readonly=True)
        try:
            # (a) SELECT via the read-only role works
            rows = ro.execute(
                f"SELECT department, ftes FROM {_TABLE_NAME} ORDER BY department"
            ).fetchall()
            assert rows == [("Fire", 890), ("Parks", 310), ("Police", 1204)]
            row = ro.execute(
                f"SELECT fy2026_budget_m, adopted FROM {_TABLE_NAME} WHERE department = 'Fire'"
            ).fetchone()
            assert row is not None
            assert float(row[0]) == pytest.approx(-3.2)
            assert row[1] == date(2026, 6, 30)
            # (b) GUARDRAIL: writes through the read-only role are rejected by Postgres
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                ro.execute(f"INSERT INTO {_TABLE_NAME} (department) VALUES ('hacked')")
            ro.rollback()
        finally:
            ro.close()

        # (c) registry row exists with the inferred column specs
        reg = db_conn.execute(
            "SELECT columns_json FROM table_registry WHERE table_name = %s", (name,)
        ).fetchall()
        assert len(reg) == 1
        assert reg[0][0] == [
            {"name": "department", "sql_type": "text"},
            {"name": "fy2026_budget_m", "sql_type": "numeric"},
            {"name": "ftes", "sql_type": "integer"},
            {"name": "adopted", "sql_type": "date"},
        ]

        # idempotent re-ingestion: same name, data replaced, one registry row
        assert store.create_normalized_table(db_conn, source_id, table) == name
        db_conn.commit()
        count_row = db_conn.execute(f"SELECT count(*) FROM {_TABLE_NAME}").fetchone()
        assert count_row is not None and count_row[0] == 3
        reg_row = db_conn.execute(
            "SELECT count(*) FROM table_registry WHERE table_name = %s", (name,)
        ).fetchone()
        assert reg_row is not None and reg_row[0] == 1
    finally:
        _cleanup(db_conn)
