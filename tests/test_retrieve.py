"""Retrieval behaviour, especially the case where the honest answer is nothing.

A vector index always returns its nearest neighbours. Ask it about a subject the
corpus has never covered and it answers anyway, at whatever distance, with no
signal that the result is irrelevant — and a language model handed those will
write a confident answer out of unrelated text. The threshold is what turns that
into an empty list, so most of these tests are about the empty case.

Scores are controlled exactly rather than approximated: documents embed to unit
vectors at named angles, so the cosine similarity between a query and a chunk is
a number the test chose in advance.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from neuronest.embed.base import Embedder, Vector
from neuronest.ingest.chunking import FixedSizeChunker
from neuronest.ingest.loaders import Document, document_id
from neuronest.retrieve import Retriever, similarity_from_distance
from neuronest.store.chroma import ChromaStore


class AngleEmbedder:
    """Embeds ``deg:N`` as the unit vector at N degrees.

    Similarity between ``deg:0`` and ``deg:60`` is then cos(60°) = 0.5, exactly,
    which makes a threshold boundary something a test can sit on deliberately
    instead of hoping a real model lands nearby.
    """

    @property
    def model_name(self) -> str:
        return "angle-v1"

    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for text in texts:
            degrees = float(text.removeprefix("deg:").strip()) if text.startswith("deg:") else 90.0
            radians = math.radians(degrees)
            vectors.append([math.cos(radians), math.sin(radians)])
        return vectors


def make_document(name: str, text: str) -> Document:
    return Document(
        doc_id=document_id(name), source_path=Path("corpus") / name, text=text, metadata={}
    )


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(AngleEmbedder(), path=tmp_path / "chroma")


def ingest(store: ChromaStore, *degrees: float) -> None:
    """Store one chunk per angle.

    Coerced to float so ``0`` and ``0.0`` produce the same text and the same
    document name; without it the chunk reads 'deg:0' or 'deg:0.0' depending on
    how the caller happened to write the literal.
    """
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=0)
    for value in degrees:
        document = make_document(f"doc-{float(value)}.md", f"deg:{float(value)}")
        store.ingest(document, chunker.chunk(document))


# --- the conversion ---------------------------------------------------------


def test_distance_becomes_similarity() -> None:
    assert similarity_from_distance(0.0) == 1.0
    assert similarity_from_distance(1.0) == 0.0
    assert similarity_from_distance(0.75) == pytest.approx(0.25)


# --- the happy path ---------------------------------------------------------


def test_the_closest_chunk_comes_back_at_rank_one(store: ChromaStore) -> None:
    ingest(store, 0, 60, 80)

    results = Retriever(store, top_k=3, score_threshold=-1.0).retrieve("deg:0")

    assert results[0].text == "deg:0.0"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_results_are_ordered_and_ranked_from_one(store: ChromaStore) -> None:
    ingest(store, 0, 60, 80)

    results = Retriever(store, top_k=3, score_threshold=-1.0).retrieve("deg:0")

    assert [item.rank for item in results] == [1, 2, 3]
    assert [item.score for item in results] == sorted((i.score for i in results), reverse=True)


def test_results_carry_their_source_and_offsets(store: ChromaStore) -> None:
    ingest(store, 0)

    result = Retriever(store, top_k=1, score_threshold=-1.0).retrieve("deg:0")[0]

    assert result.source_path.endswith("doc-0.0.md")
    assert result.doc_id == document_id("doc-0.0.md")
    assert (result.char_start, result.char_end) == (0, len("deg:0.0"))


def test_k_limits_the_result_count(store: ChromaStore) -> None:
    ingest(store, 0, 10, 20, 30)

    assert len(Retriever(store, top_k=4, score_threshold=-1.0).retrieve("deg:0", k=2)) == 2


def test_k_overrides_the_configured_default(store: ChromaStore) -> None:
    ingest(store, 0, 10, 20)

    assert len(Retriever(store, top_k=1, score_threshold=-1.0).retrieve("deg:0", k=3)) == 3


# --- the empty result, which is the point -----------------------------------


def test_a_topic_the_corpus_does_not_cover_returns_nothing(store: ChromaStore) -> None:
    """The exit criterion: an absent subject gets an empty list, not the least-bad chunk."""
    ingest(store, 0, 5, 10)

    # 89 degrees away from everything held: nearest neighbours exist, but none
    # of them are relevant, and returning one anyway is how a system invents.
    results = Retriever(store, top_k=3, score_threshold=0.25).retrieve("deg:89")

    assert results == []


def test_an_empty_store_returns_nothing(store: ChromaStore) -> None:
    assert Retriever(store, top_k=5).retrieve("deg:0") == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t "])
def test_a_blank_query_returns_nothing(store: ChromaStore, query: str) -> None:
    """A blank query has no direction to search in, so nearest-to-nothing is not an answer."""
    ingest(store, 0)

    assert Retriever(store, top_k=5, score_threshold=-1.0).retrieve(query) == []


@pytest.mark.parametrize("k", [0, -1])
def test_asking_for_no_results_returns_none(store: ChromaStore, k: int) -> None:
    ingest(store, 0)

    assert Retriever(store, top_k=5, score_threshold=-1.0).retrieve("deg:0", k=k) == []


# --- the threshold ----------------------------------------------------------


def test_chunks_below_the_threshold_are_dropped(store: ChromaStore) -> None:
    """cos(60) = 0.50 clears a 0.25 floor; cos(80) ~= 0.17 does not."""
    ingest(store, 0, 60, 80)

    results = Retriever(store, top_k=3, score_threshold=0.25).retrieve("deg:0")

    assert [item.text for item in results] == ["deg:0.0", "deg:60.0"]


def test_a_score_exactly_at_the_threshold_is_kept(store: ChromaStore) -> None:
    """The floor is inclusive, so a chunk sitting on it is not silently discarded."""
    ingest(store, 60)

    results = Retriever(store, top_k=1, score_threshold=0.5 - 1e-9).retrieve("deg:0")

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.5, abs=1e-6)


def test_ranks_are_renumbered_after_filtering(store: ChromaStore) -> None:
    """Rank is a position in what was returned, not in what the index looked at."""
    ingest(store, 0, 60, 80, 85)

    results = Retriever(store, top_k=4, score_threshold=0.25).retrieve("deg:0")

    assert [item.rank for item in results] == list(range(1, len(results) + 1))


def test_a_raised_threshold_refuses_more(store: ChromaStore) -> None:
    """The knob phase 13 tunes: strictness trades recall against false answers."""
    ingest(store, 0, 60)

    lenient = Retriever(store, top_k=2, score_threshold=0.25).retrieve("deg:0")
    strict = Retriever(store, top_k=2, score_threshold=0.9).retrieve("deg:0")

    assert len(lenient) == 2
    assert len(strict) == 1


def test_threshold_defaults_come_from_settings(store: ChromaStore) -> None:
    from neuronest.config import settings

    retriever = Retriever(store)

    assert retriever.score_threshold == settings.score_threshold
    assert retriever.top_k == settings.top_k


def test_angle_embedder_satisfies_the_protocol() -> None:
    assert isinstance(AngleEmbedder(), Embedder)
