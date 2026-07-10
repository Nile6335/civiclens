"""Graph state and the public result type for the multi-agent /ask pipeline."""

import operator
from dataclasses import asdict, dataclass, field
from typing import Annotated, TypedDict

from agents.evidence import Evidence
from retrieval.search import SearchFilters

ROUTES = ("transcript", "document", "tabular")


class AskState(TypedDict, total=False):
    question: str
    filters: SearchFilters | None
    routes: list[str]  # subset of ROUTES chosen by the supervisor
    route_source: str  # "llm" | "heuristic"
    evidence: Annotated[list[Evidence], operator.add]  # fan-out agents append
    sql: Annotated[list[str], operator.add]  # executed SQL (tabular agent)
    answer: str  # final, citations resolved


@dataclass
class AskResult:
    question: str
    answer: str
    routes: list[str] = field(default_factory=list)
    route_source: str = ""
    citations: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    sql: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def result_from_state(state: AskState) -> AskResult:
    evidence = state.get("evidence", [])
    return AskResult(
        question=state.get("question", ""),
        answer=state.get("answer", ""),
        routes=state.get("routes", []),
        route_source=state.get("route_source", ""),
        citations=[e.citation for e in evidence],
        evidence=[
            {"kind": e.kind, "text": e.text, "citation": e.citation, "score": e.score, **e.meta}
            for e in evidence
        ],
        sql=state.get("sql", []),
    )
