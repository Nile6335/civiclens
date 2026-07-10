"""Deterministic tests for the agent graph: routing, citation machinery, synthesis fallbacks."""

from agents.evidence import (
    CITATION_RE,
    NOT_FOUND,
    Evidence,
    fmt_timestamp,
    resolve_markers,
    sentences_without_citation,
    video_url_at,
)
from agents.graph import (
    _parse_routes,
    _postprocess_answer,
    extractive_answer,
    route_heuristic,
)

EV = [
    Evidence.from_video(
        "The council approved the budget amendment.", "https://youtu.be/x", 754.0, 0.9
    ),
    Evidence.from_doc("Item 12: rezoning of Elm Street parcels.", 4, 0.8, title="Agenda"),
    Evidence.from_table("columns: a | b\nrow: 1 | 2", "civic_tbl_items"),
]


def test_fmt_timestamp() -> None:
    assert fmt_timestamp(125) == "02:05"
    assert fmt_timestamp(3725) == "1:02:05"
    assert fmt_timestamp(0) == "00:00"


def test_video_url_at() -> None:
    assert video_url_at("https://youtube.com/watch?v=abc", 90) == (
        "https://youtube.com/watch?v=abc&t=90s"
    )
    assert video_url_at("https://youtu.be/abc", 90) == "https://youtu.be/abc?t=90s"


def test_citation_formats_match_regex() -> None:
    for ev in EV:
        assert CITATION_RE.fullmatch(ev.citation), ev.citation


def test_resolve_markers() -> None:
    answer = "The budget passed [E1]. Elm Street was rezoned [E2]. Bogus [E9]."
    resolved = resolve_markers(answer, EV)
    assert "[video @ 12:34](https://youtu.be/x?t=754s)" in resolved
    assert "[doc, p.4]" in resolved
    assert "[E9]" not in resolved and "[E1]" not in resolved


def test_sentences_without_citation() -> None:
    good = "The council approved the budget amendment for fiscal 2026 [doc, p.4]."
    bad = "The council approved the budget amendment without any citation whatsoever."
    assert sentences_without_citation(good) == []
    offenders = sentences_without_citation(good + " " + bad)
    assert len(offenders) == 1 and "without any citation" in offenders[0]
    assert sentences_without_citation(NOT_FOUND) == []


def test_route_heuristic() -> None:
    assert route_heuristic("What is the total number of approved items?") == ["tabular"]
    assert "tabular" in route_heuristic("How many agenda items were approved?")
    assert route_heuristic("What did the mayor say about housing?") == ["transcript"]
    assert route_heuristic("What is on page 3 of the agenda?") == ["document"]
    assert set(route_heuristic("Tell me about Elm Street.")) == {
        "transcript",
        "document",
        "tabular",
    }
    both = route_heuristic("What did the mayor say about the agenda document?")
    assert both == ["transcript", "document"]


def test_parse_routes() -> None:
    assert _parse_routes('["transcript","tabular"]') == ["transcript", "tabular"]
    assert _parse_routes('Here you go: ["document"] hope that helps') == ["document"]
    assert _parse_routes('["nonsense"]') is None
    assert _parse_routes("no json at all") is None
    assert _parse_routes("[]") is None


def test_postprocess_good_answer() -> None:
    raw = "The council approved the budget amendment [E1]. The parcels were rezoned [E2]."
    out = _postprocess_answer(raw, EV)
    assert "[video @ 12:34]" in out and "[doc, p.4]" in out
    assert sentences_without_citation(out) == []


def test_postprocess_no_markers_falls_back_to_extractive() -> None:
    raw = "The council approved lots of things and everyone was happy about the decisions."
    out = _postprocess_answer(raw, EV)
    assert out == extractive_answer(EV)
    assert CITATION_RE.search(out)
    assert sentences_without_citation(out) == []


def test_postprocess_not_found_passthrough() -> None:
    assert _postprocess_answer("Not found in the record.", EV) == NOT_FOUND
    assert _postprocess_answer("", EV) == NOT_FOUND


def test_extractive_answer_empty_evidence() -> None:
    assert extractive_answer([]) == NOT_FOUND
