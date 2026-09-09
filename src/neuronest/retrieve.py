"""Finding the passages that answer a question.

Results carry a similarity score, a rank and the document and offsets they came
from, because phase 10 has to cite its evidence rather than merely quote it.
"""

from pydantic import BaseModel, ConfigDict

from neuronest.config import settings
from neuronest.store.chroma import ChromaStore


class RetrievedChunk(BaseModel):
    """A passage that cleared the relevance bar, with where it came from.

    Carries ``source_path`` and character offsets because phase 10 cites its
    evidence: an answer has to point at the document and position it came from,
    not merely quote text back.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    source_path: str
    #: Cosine similarity in [-1, 1]. Higher is closer.
    score: float
    #: 1-based position in this result set, after filtering.
    rank: int


def similarity_from_distance(distance: float) -> float:
    """Convert Chroma's cosine distance into a similarity.

    The collection is configured for cosine and the vectors are normalised at
    encode time, so distance is ``1 - cosine_similarity``: identical text scores
    1.0, unrelated text lands near 0. Reported as similarity because a threshold
    on "how relevant" reads correctly, while a threshold on distance inverts and
    is misread the first time someone tunes it.
    """
    return 1.0 - distance


class Retriever:
    """Top-k retrieval over a store."""

    def __init__(self, store: ChromaStore, top_k: int | None = None) -> None:
        self._store = store
        self.top_k = top_k if top_k is not None else settings.top_k

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return the nearest passages to ``query``, closest first."""
        # A blank query has no direction to search in. Embedding it would still
        # produce a vector and still return the corpus's nearest neighbours to
        # nothing in particular, which is worse than answering honestly.
        if not query.strip():
            return []

        wanted = k if k is not None else self.top_k
        if wanted <= 0:
            return []

        hits = self._store.search(query, k=wanted)

        results: list[RetrievedChunk] = []
        for hit in hits:
            score = similarity_from_distance(hit.distance)
            results.append(
                RetrievedChunk(
                    chunk_id=hit.chunk.chunk_id,
                    doc_id=hit.chunk.doc_id,
                    text=hit.chunk.text,
                    char_start=hit.chunk.char_start,
                    char_end=hit.chunk.char_end,
                    source_path=hit.chunk.source_path,
                    score=score,
                    rank=len(results) + 1,
                )
            )

        return results
