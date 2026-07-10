"""Dense embedding backend (sentence-transformers), lazy-loaded and cached.

The model is configured via EMBEDDING_MODEL / EMBEDDING_DIM. BGE v1.5-style models
require an instruction prefix on queries (not on passages); bge-m3 and most others
do not.
"""

import logging
from functools import lru_cache

from common.settings import get_settings

logger = logging.getLogger(__name__)

_QUERY_PREFIXES = {
    # model-name substring -> query prefix
    "bge-small-en": "Represent this sentence for searching relevant passages: ",
    "bge-base-en": "Represent this sentence for searching relevant passages: ",
    "bge-large-en": "Represent this sentence for searching relevant passages: ",
}


class Embedder:
    """Thin wrapper: normalized embeddings, query-vs-passage asymmetry handled here."""

    def __init__(self, model_name: str, expected_dim: int) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: 100MB+ import chain

        logger.info("loading embedding model %s", model_name)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        get_dim = getattr(
            self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension
        )
        self.dim = get_dim()
        if expected_dim and self.dim != expected_dim:
            raise RuntimeError(
                f"{model_name} produces {self.dim}-d vectors but EMBEDDING_DIM={expected_dim}; "
                "fix .env (and re-migrate if the schema was created with the wrong dim)."
            )
        self._query_prefix = next(
            (p for sub, p in _QUERY_PREFIXES.items() if sub in model_name.lower()), ""
        )

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def encode_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            [self._query_prefix + text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    settings = get_settings()
    return Embedder(settings.embedding_model, settings.embedding_dim)
