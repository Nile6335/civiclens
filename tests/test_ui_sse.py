"""Unit tests for ui.sse.parse_sse_lines — pure SSE framing, no streamlit, no network."""

from ui.sse import parse_sse_lines


def test_well_formed_frame() -> None:
    lines = ["event: token", 'data: {"text": "hi"}', ""]
    assert list(parse_sse_lines(lines)) == [("token", {"text": "hi"})]


def test_keepalive_comment_lines_ignored() -> None:
    lines = [": ping", "", ": keepalive", "event: status", 'data: {"node": "supervisor"}', ""]
    assert list(parse_sse_lines(lines)) == [("status", {"node": "supervisor"})]


def test_multi_frame_stream() -> None:
    lines = [
        "event: token",
        'data: {"text": "a"}',
        "",
        "event: token",
        'data: {"text": "b"}',
        "",
        "event: result",
        'data: {"data": {"answer": "a b"}}',
        "",
    ]
    events = list(parse_sse_lines(lines))
    assert [name for name, _ in events] == ["token", "token", "result"]
    assert events[0][1] == {"text": "a"}
    assert events[2][1]["data"]["answer"] == "a b"


def test_malformed_json_frame_skipped() -> None:
    lines = [
        "event: token",
        "data: {not valid json",
        "",
        "event: token",
        'data: {"text": "ok"}',
        "",
    ]
    assert list(parse_sse_lines(lines)) == [("token", {"text": "ok"})]


def test_missing_event_name_defaults_to_message() -> None:
    lines = ['data: {"hello": 1}', ""]
    assert list(parse_sse_lines(lines)) == [("message", {"hello": 1})]


def test_final_frame_without_trailing_blank_line_still_emitted() -> None:
    lines = ["event: result", 'data: {"data": {"answer": "done"}}']
    assert list(parse_sse_lines(lines)) == [("result", {"data": {"answer": "done"}})]
