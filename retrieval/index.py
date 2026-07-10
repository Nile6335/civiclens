"""Post-ingest indexing: embed chunks without a vector, topic-tag chunks without a topic.

python -m retrieval.index
"""

import logging

from common.db import get_connection
from retrieval.embeddings import get_embedder

logger = logging.getLogger(__name__)


def tag_pending_topics() -> int:
    """Topic-tag all chunks with a NULL topic; returns how many were tagged."""
    from retrieval.topics import tag_text

    total = 0
    with get_connection() as conn:
        rows = conn.execute("SELECT id, text FROM chunks WHERE topic IS NULL").fetchall()
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET topic = %s WHERE id = %s",
                [(tag_text(text), chunk_id) for chunk_id, text in rows],
            )
        conn.commit()
        total = len(rows)
    if total:
        logger.info("tagged %d chunks", total)
    return total


def embed_pending_chunks(batch_size: int = 64) -> int:
    """Embed all chunks with a NULL embedding; returns how many were embedded."""
    embedder = get_embedder()
    total = 0
    with get_connection() as conn:
        while True:
            rows = conn.execute(
                "SELECT id, text FROM chunks WHERE embedding IS NULL ORDER BY id LIMIT %s",
                (batch_size,),
            ).fetchall()
            if not rows:
                break
            vectors = embedder.encode_passages([r[1] for r in rows])
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
                    [(str(vec), row[0]) for row, vec in zip(rows, vectors, strict=True)],
                )
            conn.commit()
            total += len(rows)
            logger.info("embedded %d chunks (total %d)", len(rows), total)
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    n_topics = tag_pending_topics()
    n = embed_pending_chunks()
    print(f"tagged {n_topics} chunks, embedded {n} chunks")


if __name__ == "__main__":
    main()
