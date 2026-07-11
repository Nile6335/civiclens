"""Minimal SSE client helpers for the CivicLens UI.

Kept free of streamlit imports so the parsing logic is unit-testable in isolation.
"""

import json
from collections.abc import Iterable, Iterator

import httpx


def parse_sse_lines(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """Parse Server-Sent-Events framing into (event_name, data_dict) tuples.

    Accumulates ``event:`` and ``data:`` fields and emits one tuple per blank-line
    frame boundary. Comment lines (starting with ``:``, e.g. keepalive pings) are
    ignored, as are frames whose data is not valid JSON. A frame without an explicit
    event name is emitted as ``"message"`` per the SSE spec.
    """
    event_name = ""
    data_lines: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:  # blank line: frame boundary
            yield from _emit(event_name, data_lines)
            event_name = ""
            data_lines = []
            continue
        if line.startswith(":"):  # comment / keepalive
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    # Be forgiving with streams that end without a trailing blank line.
    yield from _emit(event_name, data_lines)


def _emit(event_name: str, data_lines: list[str]) -> Iterator[tuple[str, dict]]:
    """Yield a single parsed frame, or nothing when it is empty or malformed."""
    if not data_lines:
        return
    try:
        payload = json.loads("\n".join(data_lines))
    except ValueError:
        return
    if isinstance(payload, dict):
        yield (event_name or "message", payload)


def stream_ask(api_base: str, payload: dict, timeout: float = 180) -> Iterator[tuple[str, dict]]:
    """POST {api_base}/ask and yield parsed SSE events as (event_name, data) tuples."""
    url = f"{api_base.rstrip('/')}/ask"
    with httpx.stream("POST", url, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        yield from parse_sse_lines(response.iter_lines())
