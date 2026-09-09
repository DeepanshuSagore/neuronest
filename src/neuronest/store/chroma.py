"""Persistent Chroma storage, keyed so that re-ingesting updates rather than duplicates.

The whole design turns on one hazard. Ingesting a document a second time without
upsert semantics does not raise, does not warn, and does not look wrong — it
just doubles the corpus. Every retrieval metric computed afterwards is then
measured against a corpus that contains each passage twice, which inflates some
numbers and deflates others with no symptom anyone would notice. Chunk ids are
deterministic precisely so this operation can be an upsert, and the duplicate
ingest test exists from the first commit rather than after the first surprise.

Collections are named per embedding model, not shared. A 384-dimension MiniLM
vector and a 1024-dimension mistral-embed vector cannot live in one Chroma
collection, and phase 15 runs several models over the same corpus. Deriving the
name from the model means switching models is a fresh index rather than an
error, and the two indexes coexist so the comparison can be re-run without
rebuilding each time.
"""

import hashlib
import re
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from neuronest.config import settings
from neuronest.embed.base import Embedder
from neuronest.ingest.chunking import Chunk
from neuronest.ingest.loaders import Document

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

    # Chroma's own aliases. Its signatures are invariant in the list element, so
    # a plain list[list[float]] is rejected where list[PyEmbedding] is wanted;
    # borrowing the alias is more honest than casting the type away.
    from chromadb.api.types import Metadata, PyEmbedding


class StoredChunk(BaseModel):
    """A chunk as it came back out of the store."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    source_path: str


class SearchHit(BaseModel):
    """A stored chunk with the distance the index reported for it.

    Distance, not score, and deliberately unshaped: turning this into a ranked,
    threshold-filtered result with an explicit empty case is phase 7's job, and
    doing any of it here would put retrieval policy inside storage.
    """

    model_config = ConfigDict(frozen=True)

    chunk: StoredChunk
    distance: float


def collection_name_for(model_name: str) -> str:
    """Derive a legal, stable Chroma collection name from an embedding model id.

    Chroma accepts 3-512 characters of ``[a-zA-Z0-9._-]`` starting and ending
    alphanumeric, which model ids like ``sentence-transformers/all-MiniLM-L6-v2``
    violate. Slugifying alone would collide two models that differ only in the
    stripped characters, so the full name is hashed onto the end.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()[:40].strip("-")
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
    return f"nn-{slug}-{digest}" if slug else f"nn-{digest}"


class ChromaStore:
    """A persistent Chroma collection scoped to one embedding model."""

    def __init__(
        self,
        embedder: Embedder,
        path: Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._embedder = embedder
        self.path = path if path is not None else settings.chroma_path
        self.collection_name = (
            collection_name
            if collection_name is not None
            else collection_name_for(embedder.model_name)
        )
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None

    def _connected(self) -> "ClientAPI":
        if self._client is None:
            import chromadb

            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.path))
        return self._client

    def _open(self) -> "Collection":
        if self._collection is None:
            self._collection = self._connected().get_or_create_collection(
                name=self.collection_name,
                # Vectors arrive already computed and already normalised, so
                # Chroma must not bring its own model. Left to default it would
                # download an ONNX MiniLM the service never uses.
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"embedding_model": self._embedder.model_name},
            )
        return self._collection

    def upsert(self, chunks: Sequence[Chunk], documents: Sequence[Document] = ()) -> None:
        """Embed and store chunks, replacing any already held under the same ids."""
        if not chunks:
            return

        source_paths = {document.doc_id: str(document.source_path) for document in documents}
        vectors: list[PyEmbedding] = list(self._embedder.embed([chunk.text for chunk in chunks]))

        metadatas: list[Metadata] = [
            {
                "doc_id": chunk.doc_id,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                # Chroma rejects None in metadata, so an unknown source is the
                # empty string rather than a missing key — every row keeps the
                # same shape and readers never branch on absence.
                "source_path": source_paths.get(chunk.doc_id, ""),
            }
            for chunk in chunks
        ]

        try:
            self._open().upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=vectors,
                documents=[chunk.text for chunk in chunks],
                metadatas=metadatas,
            )
        except Exception as exc:
            if "dimension" not in str(exc).lower():
                raise
            msg = (
                f"Collection '{self.collection_name}' holds vectors of a different width than "
                f"'{self._embedder.model_name}' produces. Embedding models cannot share a "
                f"collection; use a different chroma_path or reset() this one."
            )
            raise ValueError(msg) from exc

    def ingest(self, document: Document, chunks: Sequence[Chunk]) -> None:
        """Make the store hold exactly these chunks for this document.

        The operation ingest actually needs, and the one callers should reach
        for. Upsert alone is not enough: chunk ids encode character offsets, so
        editing a document shifts every offset after the edit and the new chunks
        arrive under new ids. Upserting them adds rows and leaves the old ones
        behind — the same silent corpus corruption as a duplicate ingest, by a
        different route, and just as invisible.

        Re-ingesting an unchanged document still touches nothing new, because
        the ids are deterministic and the stale set comes out empty.
        """
        incoming = {chunk.chunk_id for chunk in chunks}
        stale = [
            chunk_id
            for chunk_id in self._chunk_ids_for(document.doc_id)
            if chunk_id not in incoming
        ]
        if stale:
            self._open().delete(ids=stale)
        self.upsert(chunks, [document])

    def _chunk_ids_for(self, doc_id: str) -> list[str]:
        return list(self._open().get(where={"doc_id": doc_id}, include=[]).get("ids") or [])

    def count(self) -> int:
        return self._open().count()

    def document_count(self) -> int:
        """How many distinct documents the collection holds.

        Chroma indexes chunks, not documents, so this reads every chunk's
        metadata and counts distinct ids — linear in corpus size. Fine for a
        stats endpoint over tens of thousands of chunks, and worth replacing
        with a maintained counter before it is ever on a hot path.
        """
        result = self._open().get(include=["metadatas"])
        return len({(item or {}).get("doc_id") for item in result.get("metadatas") or []})

    def get(self, chunk_ids: Sequence[str]) -> list[StoredChunk]:
        if not chunk_ids:
            return []
        result = self._open().get(ids=list(chunk_ids), include=["documents", "metadatas"])
        return _rows_to_chunks(result.get("ids") or [], result, self)

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Nearest chunks to a query, closest first, with raw cosine distances."""
        if k <= 0:
            return []

        query_vector: list[PyEmbedding] = [self._embedder.embed([query])[0]]
        result = self._open().query(
            query_embeddings=query_vector,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            hits.append(
                SearchHit(
                    chunk=_to_stored_chunk(chunk_id, text, metadata),
                    distance=float(distance),
                )
            )
        return hits

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk of one document.

        Needed because upsert alone cannot shrink a document: re-ingesting an
        edited file that now yields fewer chunks would update the survivors and
        strand the rest, which is the same corpus corruption arriving by a
        different route.
        """
        self._open().delete(where={"doc_id": doc_id})

    def reset(self) -> None:
        """Drop the collection and start clean."""
        from chromadb.errors import NotFoundError

        client = self._connected()
        # Absent is the end state either way, so a collection that was never
        # created is not a failure. Narrowed to NotFoundError so a genuine
        # problem deleting an existing collection still surfaces.
        with suppress(NotFoundError):
            client.delete_collection(self.collection_name)
        self._collection = None


def _to_stored_chunk(chunk_id: str, text: str | None, metadata: Any) -> StoredChunk:
    fields: dict[str, Any] = dict(metadata or {})
    return StoredChunk(
        chunk_id=chunk_id,
        doc_id=str(fields.get("doc_id", "")),
        text=text or "",
        char_start=int(fields.get("char_start", 0)),
        char_end=int(fields.get("char_end", 0)),
        source_path=str(fields.get("source_path", "")),
    )


def _rows_to_chunks(ids: Sequence[str], result: Any, _store: ChromaStore) -> list[StoredChunk]:
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        _to_stored_chunk(chunk_id, documents[index], metadatas[index])
        for index, chunk_id in enumerate(ids)
    ]
