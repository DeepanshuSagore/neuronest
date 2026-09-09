"""The default embedder: sentence-transformers, running locally.

Local is the default for four reasons that all outlive the convenience of a
hosted API. The evaluation sweeps in phases 12-15 re-embed the corpus many
times and stay free. The test suite runs with no key and no network. The
weights are pinned, so a number published in the README still reproduces after
the vendor retires a model. And a query embeds in milliseconds instead of a
network round trip, which keeps the phase 16 retrieval latency measuring
retrieval rather than the internet.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from neuronest.config import settings
from neuronest.embed.base import Vector

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """Wraps a sentence-transformers model.

    The model is loaded on first use rather than in ``__init__``. Importing
    sentence-transformers already drags in torch; loading weights as well would
    make constructing an embedder expensive enough that tests, the CLI and the
    API would all pay for it whether or not they ever embed anything.
    """

    def __init__(self, model_name: str | None = None, batch_size: int | None = None) -> None:
        self._model_name = model_name if model_name is not None else settings.embedding_model
        self._batch_size = batch_size if batch_size is not None else settings.embedding_batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return int(self._loaded().get_embedding_dimension() or 0)

    def _loaded(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []

        encoded = self._loaded().encode(
            list(texts),
            batch_size=self._batch_size,
            # Normalised here rather than at query time so cosine similarity is
            # a dot product and every consumer gets the same convention. Chroma
            # is configured for cosine distance to match.
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in encoded]
