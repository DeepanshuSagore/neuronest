"""Store behaviour, built around the failure that has no symptom.

Ingesting a document twice without upsert semantics does not raise and does not
look wrong — it silently doubles the corpus, and every retrieval number measured
afterwards is computed against a corpus holding each passage twice. There is no
error to notice, so the test exists before the bug does.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from neuronest.embed.base import Embedder, Vector
from neuronest.ingest.chunking import Chunk, FixedSizeChunker
from neuronest.ingest.loaders import Document, document_id
from neuronest.store.chroma import ChromaStore, collection_name_for

KEYWORDS = ("alpha", "beta", "gamma")


class KeywordEmbedder:
    """Counts three keywords and normalises, so nearest-neighbour is predictable.

    A hash-based stub would give deterministic vectors but meaningless
    distances, and a real model would make these tests download weights. This
    puts the chunks in an order the test can reason about.
    """

    def __init__(self, name: str = "keyword-v1") -> None:
        self._name = name

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> int:
        return len(KEYWORDS)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for text in texts:
            lowered = text.lower()
            raw = [float(lowered.count(keyword)) for keyword in KEYWORDS]
            # A chunk mentioning none of the keywords would be a zero vector,
            # which has no direction and makes cosine distance undefined.
            magnitude = math.sqrt(sum(value * value for value in raw)) or 1.0
            vectors.append([value / magnitude for value in raw])
        return vectors


def make_document(name: str, text: str) -> Document:
    return Document(
        doc_id=document_id(name),
        source_path=Path("corpus") / name,
        text=text,
        metadata={"filename": name},
    )


def chunked(document: Document, size: int = 120, overlap: int = 20) -> list[Chunk]:
    return FixedSizeChunker(chunk_size=size, chunk_overlap=overlap).chunk(document)


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(KeywordEmbedder(), path=tmp_path / "chroma")


# --- collection naming ------------------------------------------------------


def test_collection_name_is_legal_for_chroma() -> None:
    """Chroma wants 3-512 chars of [a-zA-Z0-9._-], starting and ending alphanumeric."""
    for model in (
        "sentence-transformers/all-MiniLM-L6-v2",
        "mistral-embed",
        "BAAI/bge-small-en-v1.5",
        "...",
    ):
        name = collection_name_for(model)
        assert 3 <= len(name) <= 512
        assert name[0].isalnum() and name[-1].isalnum()
        assert all(character.isalnum() or character in "._-" for character in name)


def test_collection_name_is_stable_and_model_specific() -> None:
    assert collection_name_for("mistral-embed") == collection_name_for("mistral-embed")
    assert collection_name_for("mistral-embed") != collection_name_for("mistral-embed-dim256")


def test_models_that_slugify_alike_still_get_separate_collections() -> None:
    """Which is why the full model name is hashed onto the end rather than trusted to the slug."""
    assert collection_name_for("acme/model-v1") != collection_name_for("acme.model.v1")


def test_store_scopes_its_collection_to_the_embedding_model(tmp_path: Path) -> None:
    """384-dim and 1024-dim vectors cannot share a collection, and phase 15 runs both."""
    mini = ChromaStore(KeywordEmbedder("mini"), path=tmp_path / "chroma")
    large = ChromaStore(KeywordEmbedder("large"), path=tmp_path / "chroma")

    assert mini.collection_name != large.collection_name


# --- the trap ---------------------------------------------------------------


def test_reingesting_a_document_does_not_duplicate_chunks(store: ChromaStore) -> None:
    """The exit criterion. Ingest twice, chunk count unchanged."""
    document = make_document("paper.md", "alpha beta gamma. " * 40)
    chunks = chunked(document)
    assert len(chunks) > 1

    store.upsert(chunks, [document])
    after_first = store.count()

    store.upsert(chunks, [document])

    assert after_first == len(chunks)
    assert store.count() == after_first


def test_ingesting_an_edited_document_replaces_it(store: ChromaStore) -> None:
    """Edited file, same path: the old text is replaced, not accumulated."""
    first = make_document("notes.md", "alpha alpha alpha")
    store.ingest(first, chunked(first))

    edited = make_document("notes.md", "beta beta beta")
    store.ingest(edited, chunked(edited))

    stored = store.get([chunked(edited)[0].chunk_id])
    assert store.count() == 1
    assert stored[0].text == "beta beta beta"


def test_upsert_alone_cannot_replace_an_edited_document(store: ChromaStore) -> None:
    """Why ingest() exists, pinned so nobody 'simplifies' it back to a bare upsert.

    Chunk ids encode character offsets. An edit shifts them, so the new chunks
    arrive under new ids and upsert adds rather than replaces.
    """
    first = make_document("notes.md", "alpha alpha alpha")
    store.upsert(chunked(first), [first])

    edited = make_document("notes.md", "beta beta beta")
    store.upsert(chunked(edited), [edited])

    assert store.count() == 2


def test_ingesting_a_shorter_edit_leaves_no_orphans(store: ChromaStore) -> None:
    """A document that shrinks must not strand the chunks it no longer has."""
    long_version = make_document("notes.md", "alpha beta gamma. " * 40)
    store.ingest(long_version, chunked(long_version))
    assert store.count() > 1

    short_version = make_document("notes.md", "alpha only now")
    store.ingest(short_version, chunked(short_version))

    assert store.count() == 1


def test_reingesting_an_unchanged_document_through_ingest_is_stable(store: ChromaStore) -> None:
    document = make_document("paper.md", "alpha beta gamma. " * 40)
    chunks = chunked(document)

    store.ingest(document, chunks)
    store.ingest(document, chunks)

    assert store.count() == len(chunks)


def test_ingest_leaves_other_documents_alone(store: ChromaStore) -> None:
    keeper = make_document("keeper.md", "gamma gamma gamma")
    store.ingest(keeper, chunked(keeper))

    edited = make_document("notes.md", "alpha alpha")
    store.ingest(edited, chunked(edited))
    edited_again = make_document("notes.md", "beta")
    store.ingest(edited_again, chunked(edited_again))

    assert store.count() == 2
    assert store.get([chunked(keeper)[0].chunk_id])[0].text == "gamma gamma gamma"


# --- storing and reading back -----------------------------------------------


def test_chunks_round_trip_with_their_offsets_and_source(store: ChromaStore) -> None:
    document = make_document("paper.md", "alpha beta gamma. " * 20)
    chunks = chunked(document)

    store.upsert(chunks, [document])
    stored = {item.chunk_id: item for item in store.get([c.chunk_id for c in chunks])}

    assert len(stored) == len(chunks)
    for chunk in chunks:
        held = stored[chunk.chunk_id]
        assert held.text == chunk.text
        assert (held.char_start, held.char_end) == (chunk.char_start, chunk.char_end)
        assert held.doc_id == document.doc_id
        assert held.source_path.endswith("paper.md")


def test_source_path_is_empty_rather_than_missing_when_unknown(store: ChromaStore) -> None:
    """Chroma rejects None in metadata, so every row keeps the same shape."""
    document = make_document("orphan.md", "alpha beta")

    store.upsert(chunked(document))

    assert store.get([chunked(document)[0].chunk_id])[0].source_path == ""


def test_upserting_nothing_is_a_noop(store: ChromaStore) -> None:
    store.upsert([])

    assert store.count() == 0


def test_getting_nothing_returns_nothing(store: ChromaStore) -> None:
    assert store.get([]) == []


# --- search -----------------------------------------------------------------


def test_search_returns_the_nearest_chunk_first(store: ChromaStore) -> None:
    alpha = make_document("alpha.md", "alpha alpha alpha")
    beta = make_document("beta.md", "beta beta beta")
    store.upsert(chunked(alpha), [alpha])
    store.upsert(chunked(beta), [beta])

    hits = store.search("alpha", k=2)

    assert len(hits) == 2
    assert hits[0].chunk.text == "alpha alpha alpha"
    assert hits[0].distance < hits[1].distance


def test_search_respects_k(store: ChromaStore) -> None:
    document = make_document("paper.md", "alpha beta gamma. " * 40)
    store.upsert(chunked(document), [document])

    assert len(store.search("alpha", k=2)) == 2


def test_search_on_an_empty_store_returns_nothing(store: ChromaStore) -> None:
    assert store.search("alpha", k=5) == []


def test_search_with_non_positive_k_returns_nothing(store: ChromaStore) -> None:
    assert store.search("alpha", k=0) == []


# --- reset ------------------------------------------------------------------


def test_reset_empties_the_collection(store: ChromaStore) -> None:
    document = make_document("paper.md", "alpha beta gamma. " * 20)
    store.upsert(chunked(document), [document])
    assert store.count() > 0

    store.reset()

    assert store.count() == 0


def test_reset_on_a_fresh_store_is_not_an_error(store: ChromaStore) -> None:
    store.reset()

    assert store.count() == 0


def test_store_persists_across_instances(tmp_path: Path) -> None:
    """The index is on disk; a restart must not lose the corpus."""
    document = make_document("paper.md", "alpha beta gamma. " * 20)

    first = ChromaStore(KeywordEmbedder(), path=tmp_path / "chroma")
    first.upsert(chunked(document), [document])
    expected = first.count()

    reopened = ChromaStore(KeywordEmbedder(), path=tmp_path / "chroma")

    assert reopened.count() == expected


# --- the protocol the store depends on --------------------------------------


def test_keyword_embedder_satisfies_the_protocol() -> None:
    assert isinstance(KeywordEmbedder(), Embedder)
