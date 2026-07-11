"""RAGAS metrics over end-to-end pipeline answers, using the local LLM and embeddings.

No OpenAI key is ever required: the judge LLM comes from common.llm.get_chat_model()
(Ollama by default) and embeddings delegate to retrieval.embeddings.get_embedder().
The installed ragas 0.2.x accepts legacy v1 column names (question/answer/contexts/
ground_truth) and remaps them internally (ground_truth -> reference).

All heavy imports (ragas, datasets, agents.graph, the embedding model) are deferred so
importing this module never loads models or touches the database.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings

from evals.dataset import GoldenItem

if TYPE_CHECKING:
    from retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def collect_answers(
    items: list[GoldenItem],
    ask_fn: Callable[[str], Any] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Run the pipeline on each golden item and return ragas-ready records.

    ask_fn defaults to agents.graph.ask and must return an AskResult-shaped object
    (.answer str, .evidence list of dicts with a "text" key). A failing ask_fn records
    the item with an empty answer and placeholder context instead of aborting the run.
    """
    if ask_fn is None:
        from agents.graph import ask  # lazy: pulls in the langgraph + LLM stack

        ask_fn = ask
    records: list[dict] = []
    for item in items[:limit]:
        try:
            result = ask_fn(item.question)
            answer = result.answer
            contexts = [e["text"] for e in result.evidence] or [""]
        except Exception:
            logger.exception("ask_fn failed for item %s; recording empty answer", item.id)
            answer, contexts = "", [""]
        records.append(
            {
                "id": item.id,
                "question": item.question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item.answer,
            }
        )
    return records


class LocalEmbeddings(Embeddings):
    """langchain Embeddings adapter over the repo's sentence-transformers embedder.

    Lazy: the underlying model is loaded on the first embed call, never at import or
    construction time.
    """

    def _embedder(self) -> "Embedder":
        from retrieval.embeddings import get_embedder

        return get_embedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder().encode_passages(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder().encode_query(text)


def ragas_scores(records: list[dict], max_workers: int = 2) -> dict:
    """Evaluate collected records with RAGAS and reduce to per-metric NaN-aware means.

    Returns {"faithfulness": float, "answer_relevancy": float, "context_precision":
    float, "context_recall": float, "n": int, "per_metric_n": {metric: non-NaN rows}}.
    Raises RuntimeError on any hard ragas failure.
    """
    if not records:
        raise RuntimeError("ragas_scores requires at least one record")

    import numpy as np
    from datasets import Dataset

    from common.llm import get_chat_model

    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": list(r["contexts"]) or [""],  # empty contexts must not crash
                "ground_truth": r["ground_truth"],
            }
            for r in records
        ]
    )

    try:
        from ragas import RunConfig, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        run_config = RunConfig(timeout=120, max_workers=max_workers)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=LangchainLLMWrapper(get_chat_model()),
            embeddings=LangchainEmbeddingsWrapper(LocalEmbeddings()),
            run_config=run_config,
            show_progress=False,
        )
        frame = result.to_pandas()
    except Exception as exc:
        raise RuntimeError(f"ragas evaluation failed: {exc}") from exc

    scores: dict[str, Any] = {}
    per_metric_n: dict[str, int] = {}
    for name in METRIC_NAMES:
        if name in frame.columns:
            column = frame[name].to_numpy(dtype=float)
        else:
            column = np.array([], dtype=float)
        n_valid = int(np.count_nonzero(~np.isnan(column)))
        per_metric_n[name] = n_valid
        scores[name] = float(np.nanmean(column)) if n_valid else float("nan")
    scores["n"] = len(records)
    scores["per_metric_n"] = per_metric_n
    return scores
