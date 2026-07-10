"""Normalize extracted budget/vote tables (RawTable) into typed NormalizedTables.

Headers become snake_case SQL identifiers, column types are inferred from the cell
values (integer/numeric/date/text), and cells are coerced so the tabular agent can
query the resulting civic_tbl_* tables with plain SQL.
"""

import csv
import re
from datetime import date, datetime
from pathlib import Path

from ingestion.models import ColumnSpec, NormalizedTable, RawTable

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y")
_MAX_IDENT_LEN = 58


def snake_case_ident(name: str, fallback: str = "col") -> str:
    """Turn a free-form header into a safe snake_case SQL identifier."""
    ident = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not ident:
        ident = fallback
    elif ident[0].isdigit():
        ident = f"{fallback}_{ident}"
    return ident[:_MAX_IDENT_LEN]


def dedupe_idents(names: list[str]) -> list[str]:
    """Make identifiers unique by appending _2, _3, ... on collision."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            out.append(name)
            continue
        n = seen[name] + 1
        candidate = f"{name}_{n}"
        while candidate in seen:
            n += 1
            candidate = f"{name}_{n}"
        seen[name] = n
        seen[candidate] = 1
        out.append(candidate)
    return out


def _clean_numeric(value: str) -> str:
    """Strip $/,/% and turn parenthesized negatives like (1,234.56) into -1234.56."""
    cleaned = value.strip().replace("$", "").replace(",", "").replace("%", "").strip()
    if len(cleaned) > 2 and cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    return cleaned


def _try_int(value: str) -> int | None:
    cleaned = value.strip().replace(",", "")
    return int(cleaned) if _INT_RE.match(cleaned) else None


def _try_float(value: str) -> float | None:
    cleaned = _clean_numeric(value)
    return float(cleaned) if _FLOAT_RE.match(cleaned) else None


def _try_date(value: str) -> date | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def infer_column_type(values: list[str]) -> str:
    """Infer integer/numeric/date/text from non-empty cell values (text by default)."""
    non_empty = [v.strip() for v in values if v is not None and v.strip()]
    if not non_empty:
        return "text"
    if all(_try_int(v) is not None for v in non_empty):
        return "integer"
    if all(_try_float(v) is not None for v in non_empty):
        return "numeric"
    if all(_try_date(v) is not None for v in non_empty):
        return "date"
    return "text"


def coerce(value: str | None, sql_type: str) -> object | None:
    """Coerce one cell to its SQL type using the same cleaning as inference."""
    if value is None or not value.strip():
        return None
    stripped = value.strip()
    if sql_type == "text":
        return stripped
    parsed: object | None
    if sql_type == "integer":
        parsed = _try_int(stripped)
    elif sql_type == "numeric":
        parsed = _try_float(stripped)
    elif sql_type == "date":
        parsed = _try_date(stripped)
    else:
        raise ValueError(f"unknown sql_type: {sql_type!r}")
    if parsed is None:
        raise ValueError(f"cannot coerce {value!r} to {sql_type}")
    return parsed


def normalize_raw_table(raw: RawTable, slug: str, description: str) -> NormalizedTable:
    """Normalize an extracted table: clean headers, square up rows, type the columns."""
    if not _SLUG_RE.match(slug):
        raise ValueError(f"invalid table slug: {slug!r} (must match ^[a-z0-9_]+$)")
    names = dedupe_idents([snake_case_ident(h) for h in raw.header])
    ncols = len(names)
    rows: list[list[str]] = []
    for row in raw.rows:
        if all(not (cell or "").strip() for cell in row):
            continue
        cells = row[:ncols] if len(row) > ncols else row + [""] * (ncols - len(row))
        rows.append(cells)
    columns = [
        ColumnSpec(name=names[i], sql_type=infer_column_type([r[i] for r in rows]))
        for i in range(ncols)
    ]
    coerced = [[coerce(r[i], columns[i].sql_type) for i in range(ncols)] for r in rows]
    return NormalizedTable(slug=slug, description=description, columns=columns, rows=coerced)


def load_csv_table(path: Path, slug: str, description: str) -> NormalizedTable:
    """Read a CSV (first row = header) and normalize it into a NormalizedTable."""
    with path.open(newline="", encoding="utf-8") as f:
        records = list(csv.reader(f))
    if not records:
        raise ValueError(f"empty CSV file: {path}")
    return normalize_raw_table(RawTable(header=records[0], rows=records[1:]), slug, description)
