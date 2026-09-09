"""Finding the passages that answer a question, or deciding that none do.

The second half of that sentence is the point. A vector index always returns
something: ask it about a subject the corpus has never heard of and it will
hand back its nearest neighbours anyway, at whatever distance, with no signal
that they are irrelevant. Pass those to a language model and it will write a
confident answer out of unrelated text, because that is what it was given.

So an empty result is a value here, not an error and not a failure. Retrieval
returns what cleared the bar; when nothing clears it, that is the honest answer
and phase 10 refuses on it without ever calling the model.
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
    """Top-k retrieval over a store, filtered by a relevance floor."""

    def __init__(
        self,
        store: ChromaStore,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self._store = store
        self.top_k = top_k if top_k is not None else settings.top_k
        self.score_threshold = (
            score_threshold if score_threshold is not None else settings.score_threshold
        )

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return the passages relevant to ``query``, closest first.

        An empty list means nothing in the corpus was relevant enough — not that
        something went wrong.
        """
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
            if score < self.score_threshold:
                # Hits arrive sorted by distance, so the first one to fall below
                # the floor means every hit after it does too.
                break
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
