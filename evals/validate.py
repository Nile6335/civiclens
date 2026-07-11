"""Validation pass over the generated golden dataset.

    python -m evals.validate

Each LLM-generated pair is verified on: (a) answerable solely from its cited span,
(b) unambiguous, (c) non-trivial. Failures are discarded (kept in
golden_rejected.json for transparency); the acceptance rate is logged. Table items are
generated programmatically from the tables themselves — deterministically correct by
construction — and are auto-passed with a note.

Judge design (measured, not assumed): the local qwen2.5:1.5b judge answers "no" on
messy spoken-transcript spans even when the answer is verifiably present (traced on
real rejects), so with the ollama backend criterion (a) is verified PROGRAMMATICALLY
(strict answer-support in the span) and the LLM judges only the question-quality
criteria (b) and (c), which are short, question-only binary tasks it handles reliably.
With LLM_BACKEND=anthropic all three criteria are LLM-judged.
"""

import json
import logging
import re

from common.llm import get_chat_model
from evals.dataset import DATASET_PATH, GoldenItem, load_dataset, save_dataset

logger = logging.getLogger(__name__)

REJECTED_PATH = DATASET_PATH.with_name("golden_rejected.json")

# Small local models fail at multi-criteria rubrics (they anchor on the example's
# verdict), so each criterion is a separate BINARY check with one pass and one fail
# example. Checks short-circuit on the first failure.

_ANSWERABLE_PROMPT = """Does the source text contain the answer to the question, and is \
the proposed answer correct according to the text? Reply only "yes" or "no".

Example 1:
Text: "The library renovation will cost $2.4 million and finish in October."
Question: How much will the library renovation cost? | Proposed answer: $2.4 million
Reply: yes

Example 2:
Text: "The library renovation will cost $2.4 million and finish in October."
Question: Who is the city treasurer? | Proposed answer: John Smith
Reply: no

Now:
Text: \"\"\"{spans}\"\"\"
Question: {question} | Proposed answer: {answer}
Reply:"""

_UNAMBIGUOUS_PROMPT = """Does this question have exactly one clear, specific answer \
(not a matter of opinion, not vague)? Reply only "yes" or "no".

Example 1:
Question: What time does the March 3 council meeting start?
Reply: yes

Example 2:
Question: How do residents feel about the city's direction?
Reply: no

Example 3:
Question: At the mesa city council meeting of 2026-04-06, who received the \
environmental excellence award?
Reply: yes

Now:
Question: {question}
Reply:"""

_TRIVIAL_PROMPT = """Could most people answer this question correctly WITHOUT any source \
document, using only common knowledge? Reply only "yes" or "no".

Example 1:
Question: How many days are in a week?
Reply: yes

Example 2:
Question: What is the matter file number of agenda item 5 at the Mesa council meeting?
Reply: no

Now:
Question: {question}
Reply:"""

_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _full_span_texts(item: GoldenItem) -> str:
    """Full chunk texts from the DB (snippets are truncated); snippet fallback."""
    try:
        from common.db import get_connection

        texts = []
        with get_connection() as conn:
            for s in item.spans:
                row = conn.execute(
                    """
                    SELECT c.text FROM chunks c JOIN sources s2 ON s2.id = c.source_id
                    WHERE s2.city = %s AND s2.source_type = %s AND s2.meeting_id = %s
                      AND c.chunk_index = %s
                    """,
                    (s.city, s.source_type, s.meeting_id, s.chunk_index),
                ).fetchone()
                texts.append(row[0] if row else s.text_snippet)
        return "\n---\n".join(texts)
    except Exception:  # DB down: judge against what we stored
        return "\n---\n".join(s.text_snippet for s in item.spans)


def _focus_window(span_text: str, answer: str, width: int = 700) -> str:
    """Trim the span to the region around the answer: weak judges lose the needle in
    ~1000-char spans. Falls back to the span head when the answer isn't found."""
    lowered = span_text.lower()
    idx = lowered.find(answer.lower()[:80])
    if idx == -1:
        for word in re.findall(r"[a-z0-9]{4,}", answer.lower()):
            idx = lowered.find(word)
            if idx != -1:
                break
    if idx == -1:
        return span_text[: width * 2]
    start = max(0, idx - width // 2)
    return span_text[start : idx + width]


def _ask_yes_no(prompt: str) -> bool | None:
    """Run one binary check; None on an unparseable/failed call."""
    try:
        raw = str(get_chat_model().invoke(prompt).content)
    except Exception:
        return None
    match = _YESNO_RE.search(raw)
    if not match:
        return None
    return match.group(1).lower() == "yes"


def _answer_supported(answer: str, span_text: str) -> bool:
    """Strict programmatic span-support: exact substring for short answers, else >=70%
    of content words present."""
    a = " ".join(answer.lower().split())
    s = " ".join(span_text.lower().split())
    if len(a) <= 40 and a in s:
        return True
    words = [w for w in re.findall(r"[a-z0-9]+", a) if len(w) > 2]
    if not words:
        return a in s
    return sum(w in s for w in words) / len(words) >= 0.7


def _llm_judges_answerable() -> bool:
    from common.settings import get_settings

    settings = get_settings()
    return settings.llm_backend == "anthropic" and bool(settings.anthropic_api_key)


def _judge(item: GoldenItem) -> tuple[bool, str]:
    full_text = _full_span_texts(item)
    if _llm_judges_answerable():
        spans_text = _focus_window(full_text, item.answer)
        answerable = _ask_yes_no(
            _ANSWERABLE_PROMPT.format(spans=spans_text, question=item.question, answer=item.answer)
        )
        if answerable is not True:
            return False, "failed: answerable" if answerable is False else "judge call failed"
    elif not _answer_supported(item.answer, full_text):
        return False, "failed: answerable (programmatic span-support)"
    unambiguous = _ask_yes_no(_UNAMBIGUOUS_PROMPT.format(question=item.question))
    if unambiguous is not True:
        return False, "failed: unambiguous" if unambiguous is False else "judge call failed"
    trivial = _ask_yes_no(_TRIVIAL_PROMPT.format(question=item.question))
    if trivial is not False:
        return False, "failed: nontrivial" if trivial is True else "judge call failed"
    return True, "ok"


def validate(items: list[GoldenItem]) -> tuple[list[GoldenItem], list[GoldenItem]]:
    kept: list[GoldenItem] = []
    rejected: list[GoldenItem] = []
    for i, item in enumerate(items, start=1):
        if item.source_type == "table" and not item.spans:
            item.validated = True
            item.judge_notes = "deterministic (generated from the table itself)"
            kept.append(item)
            continue
        passed, notes = _judge(item)
        item.validated = passed
        item.judge_notes = notes
        (kept if passed else rejected).append(item)
        logger.info("[%d/%d] %s — %s", i, len(items), "PASS" if passed else "REJECT", item.id)
    return kept, rejected


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    items = load_dataset()
    judged = [i for i in items if not (i.source_type == "table" and not i.spans)]
    kept, rejected = validate(items)
    save_dataset(kept)
    REJECTED_PATH.write_text(
        json.dumps([i.to_dict() for i in rejected], indent=1), encoding="utf-8"
    )
    rate = (len(judged) - len(rejected)) / len(judged) if judged else 1.0
    print(
        f"kept {len(kept)}/{len(items)} items "
        f"(LLM-judged acceptance rate {rate:.0%} over {len(judged)} judged; "
        f"{len(items) - len(judged)} deterministic table items auto-passed)"
    )
    print(f"rejects saved to {REJECTED_PATH}")


if __name__ == "__main__":
    main()
