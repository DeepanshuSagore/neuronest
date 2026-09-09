"""What the endpoints are built out of, assembled in one place.

The service holds exactly one embedder, one store and one retriever for its
lifetime. That is not a convenience: the embedder owns several hundred megabytes
of model weights, and building one per request would make every query pay a
model load.

Everything is built on first use rather than in the constructor. Importing this
module — which uvicorn does to find the app — must not open a database, create a
cache directory or pull in langchain. Construction is cheap; the first request
that needs a thing pays for it.

Overrides are constructor arguments so the tests can hand the app a stub
embedder and a temporary index and drive the whole HTTP surface offline. An app
that can only be tested against real weights is an app whose tests get skipped.
"""

from functools import cached_property
from pathlib import Path

from neuronest.config import settings
from neuronest.embed.base import Embedder
from neuronest.ingest.chunking import Chunker
from neuronest.retrieve import Retriever
from neuronest.store.chroma import ChromaStore


class Services:
    """The long-lived objects an endpoint needs."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        chunker: Chunker | None = None,
        chroma_path: Path | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self._embedder = embedder
        self._chunker = chunker
        self._chroma_path = chroma_path if chroma_path is not None else settings.chroma_path
        self._cache_path = cache_path

    @cached_property
    def embedder(self) -> Embedder:
        if self._embedder is not None:
            return self._embedder

        from neuronest.embed.cache import CachingEmbedder, EmbeddingCache
        from neuronest.embed.local import LocalEmbedder

        cache_path = self._cache_path or settings.embedding_cache_path / "embeddings.db"
        return CachingEmbedder(LocalEmbedder(), EmbeddingCache(cache_path))

    @cached_property
    def chunker(self) -> Chunker:
        if self._chunker is not None:
            return self._chunker

        from neuronest.ingest.chunking import FixedSizeChunker

        return FixedSizeChunker()

    @cached_property
    def store(self) -> ChromaStore:
        return ChromaStore(self.embedder, path=self._chroma_path)

    @cached_property
    def retriever(self) -> Retriever:
        return Retriever(self.store)
