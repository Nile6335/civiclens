"""CivicLens Streamlit UI: streaming Q&A over council meetings with cited evidence."""

import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # streamlit run ui/app.py puts ui/ on sys.path, not the root
    sys.path.insert(0, str(_REPO_ROOT))

from agents.evidence import CITATION_RE  # noqa: E402  (dependency-light, stdlib only)
from ui.sse import stream_ask  # noqa: E402

st.set_page_config(page_title="CivicLens", page_icon="🏛️", layout="wide")

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
LANGFUSE_URL: str = "http://localhost:3002"

FALLBACK_EXAMPLES: tuple[str, ...] = (
    "Who was excused from the Mesa city council meeting on April 6, 2026?",
    "What items are on the consent agenda for the April 6 Mesa meeting?",
    "What awards were announced at the Mesa city council meeting on 2026-04-06?",
)

CITY_OPTIONS: tuple[str, ...] = ("All", "mesa", "seattle")
SOURCE_TYPE_OPTIONS: tuple[str, ...] = ("All", "transcript", "pdf", "table")
TOPIC_OPTIONS: tuple[str, ...] = (
    "All",
    "zoning",
    "budget",
    "public safety",
    "transportation",
    "housing",
    "other",
)

_VIDEO_CITATION_RE = re.compile(r"\[video @ ([\d:]+)\]\(([^)]+)\)")
_T_SUFFIX_RE = re.compile(r"[?&]t=\d+s?$")
_T_PARAM_RE = re.compile(r"[?&]t=(\d+)s?")
_MD_LINK_RE = re.compile(r"\]\([^)]*\)")


def clean_video_url(url: str) -> str:
    """Strip a trailing &t=..s / ?t=..s suffix so st.video gets a plain URL."""
    return _T_SUFFIX_RE.sub("", url)


def video_citations(answer: str) -> list[tuple[str, str, int]]:
    """Unique (mm:ss label, clean url, seconds) triples for video citations in the answer."""
    found: list[tuple[str, str, int]] = []
    seen: set[tuple[str, int]] = set()
    for match in CITATION_RE.finditer(answer):
        video = _VIDEO_CITATION_RE.fullmatch(match.group(0))
        if not video:
            continue
        label, url = video.group(1), video.group(2)
        t_param = _T_PARAM_RE.search(url)
        seconds = int(t_param.group(1)) if t_param else 0
        key = (clean_video_url(url), seconds)
        if key not in seen:
            seen.add(key)
            found.append((label, key[0], seconds))
    return found


@st.cache_data(ttl=60, show_spinner=False)
def fetch_examples(api_base: str) -> list[str]:
    """Example questions from GET /examples, falling back to a canned Mesa list."""
    try:
        response = httpx.get(f"{api_base}/examples", timeout=2.0)
        response.raise_for_status()
        raw = response.json().get("examples", [])
        questions = [
            item["question"]
            for item in raw
            if isinstance(item, dict) and isinstance(item.get("question"), str)
        ]
        if questions:
            return questions
    except (httpx.HTTPError, ValueError):
        pass
    return list(FALLBACK_EXAMPLES)


def render_sidebar() -> dict[str, str]:
    """Example-question buttons plus filters; returns the non-'All' filter payload."""
    with st.sidebar:
        st.header("Example questions")
        for i, question in enumerate(fetch_examples(API_BASE_URL)):
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state["question"] = question
                st.session_state["auto_run"] = True
        st.divider()
        st.header("Filters")
        st.selectbox("City", CITY_OPTIONS, key="filter_city")
        st.selectbox("Source type", SOURCE_TYPE_OPTIONS, key="filter_source_type")
        st.selectbox("Topic", TOPIC_OPTIONS, key="filter_topic")
        st.divider()
        st.caption(
            f"[🎙️ Voice mode]({API_BASE_URL}/voice-demo) · "
            f"[API docs]({API_BASE_URL}/docs) · [Langfuse]({LANGFUSE_URL})"
        )
    filters: dict[str, str] = {}
    for name in ("city", "source_type", "topic"):
        value = st.session_state.get(f"filter_{name}", "All")
        if value != "All":
            filters[name] = value
    return filters


def _status_text(data: dict) -> str:
    routes = data.get("routes") or []
    if routes:
        return "consulted: " + ", ".join(str(r) for r in routes)
    return f"running: {data.get('node', 'pipeline')}…"


def _show_unreachable_error() -> None:
    st.error(
        "Could not reach the CivicLens API — is the stack running? "
        "Start everything with `make demo` and try again."
    )


def run_ask(question: str, filters: dict[str, str]) -> None:
    """Stream one /ask call, updating status + answer preview; store the final result."""
    payload: dict[str, Any] = {"question": question, "stream": True, **filters}
    st.session_state.pop("result", None)
    st.session_state.pop("active_video", None)
    status_line = st.empty()
    preview = st.empty()
    status_line.caption("contacting the CivicLens pipeline…")
    streamed = ""
    try:
        for event, data in stream_ask(API_BASE_URL, payload):
            if event == "status":
                status_line.caption(_status_text(data))
            elif event == "token":
                streamed += str(data.get("text", ""))
                preview.markdown(streamed + " ▌")
            elif event == "result":
                st.session_state["result"] = data.get("data", {})
                status_line.empty()
                preview.empty()  # the stored result is re-rendered below, citations resolved
            elif event == "error":
                status_line.empty()
                preview.empty()
                st.error(
                    "The pipeline reported an error: "
                    f"{data.get('message', 'unknown error')}. "
                    "If parts of the stack are down, restart with `make demo`."
                )
                return
    except httpx.HTTPError:
        status_line.empty()
        preview.empty()
        _show_unreachable_error()


def render_citation_chips(answer: str) -> None:
    """'Jump to citation' chips: one ▶ button per video citation in the final answer."""
    citations = video_citations(answer)
    if not citations:
        return
    st.caption("Jump to citation")
    columns = st.columns(min(len(citations), 6))
    for i, (label, url, seconds) in enumerate(citations):
        with columns[i % len(columns)]:
            if st.button(f"▶ {label}", key=f"jump_{i}"):
                st.session_state["active_video"] = (url, seconds)


def render_active_video() -> None:
    """Embedded player near the answer, seeked to the clicked citation's timestamp."""
    active = st.session_state.get("active_video")
    if not active:
        return
    url, seconds = active
    st.video(url, start_time=int(seconds))


def _evidence_label(item: dict) -> str:
    parts = [_MD_LINK_RE.sub("]", str(item.get("citation", "evidence")))]
    if item.get("title"):
        parts.append(str(item["title"]))
    if item.get("meeting_date"):
        parts.append(str(item["meeting_date"]))
    return " — ".join(parts)


def _render_evidence_body(item: dict) -> None:
    kind = item.get("kind")
    if kind == "video":
        url = clean_video_url(str(item.get("url", "")))
        if url:
            st.video(url, start_time=int(float(item.get("t_start") or 0)))
        st.markdown(str(item.get("text", "")))
    elif kind == "doc":
        st.markdown(str(item.get("text", "")))
        if item.get("page_no") is not None:
            st.caption(f"page {item['page_no']}")
        if item.get("url"):
            st.markdown(f"[Source PDF]({item['url']})")
    elif kind == "table":
        st.code(str(item.get("text", "")))
        if item.get("sql"):
            st.code(str(item["sql"]), language="sql")
    else:
        st.write(str(item.get("text", "")))


def render_evidence(evidence: list[dict]) -> None:
    if not evidence:
        return
    st.subheader("Evidence")
    for item in evidence:
        with st.expander(_evidence_label(item)):
            _render_evidence_body(item)


def render_result(result: dict) -> None:
    """Final answer + citation chips + seeked player + evidence panel (rerun-safe)."""
    answer = str(result.get("answer", ""))
    st.markdown(answer)
    render_citation_chips(answer)
    render_active_video()
    render_evidence(result.get("evidence", []))


def main() -> None:
    filters = render_sidebar()
    st.title("🏛️ CivicLens")
    st.caption(
        "Ask questions about Mesa and Seattle city-council meetings — answers cite "
        "video timestamps, agenda pages, and data tables."
    )
    st.text_input(
        "Your question",
        key="question",
        placeholder="e.g. Who was excused from the April 6 Mesa council meeting?",
    )
    ask_clicked = st.button("Ask", type="primary")
    auto_run = bool(st.session_state.pop("auto_run", False))
    question = str(st.session_state.get("question", "")).strip()
    if (ask_clicked or auto_run) and question:
        run_ask(question, filters)
    elif ask_clicked:
        st.warning("Type a question first.")
    result = st.session_state.get("result")
    if result:
        render_result(result)


main()
