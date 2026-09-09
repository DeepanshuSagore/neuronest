"""Chunking behaviour, with determinism treated as the property that matters.

Phase 11 labels a hundred evaluation questions by chunk id. Every assertion
about stability here is protecting that set from silently rotting the next time
the corpus is re-ingested.
"""

from itertools import pairwise
from pathlib import Path

import pytest

from neuronest.ingest.chunking import (
    Chunker,
    FixedSizeChunker,
    chunk_documents,
    chunk_id,
)
from neuronest.ingest.loaders import Document, document_id

PROSE = (
    "Retrieval is not generation. A retrieval failure and a generation failure look "
    "identical from the outside, which is why they have to be measured apart.\n\n"
    "Most systems report a single number and call it accuracy. That number cannot "
    "distinguish a model that invented an answer from one that was handed the wrong "
    "passage in the first place.\n\n"
    "The fix is unglamorous: label the passage that actually contains each answer, "
    "before any tuning happens, and score the two stages separately against it.\n\n"
    "Refusal is the part almost everyone skips. A system that never declines is not "
    "confident, it is unfalsifiable."
)


def make_document(text: str = PROSE, name: str = "notes.md") -> Document:
    return Document(
        doc_id=document_id(name),
        source_path=Path("corpus") / name,
        text=text,
        metadata={"filename": name},
    )


# --- the protocol -----------------------------------------------------------


def test_fixed_size_chunker_satisfies_the_protocol() -> None:
    """Phase 14 adds a second strategy; this is what says the seam is real."""
    assert isinstance(FixedSizeChunker(chunk_size=200, chunk_overlap=40), Chunker)


def test_fingerprint_describes_the_configuration() -> None:
    assert FixedSizeChunker(chunk_size=250, chunk_overlap=50).fingerprint == "fixed-250-50"


# --- offsets ----------------------------------------------------------------


def test_offsets_slice_back_to_the_source_text() -> None:
    """A citation points into the document, so the offsets have to be exact."""
    document = make_document()

    for chunk in FixedSizeChunker(chunk_size=200, chunk_overlap=40).chunk(document):
        assert document.text[chunk.char_start : chunk.char_end] == chunk.text


def test_offsets_advance_and_chunks_carry_their_document() -> None:
    document = make_document()

    chunks = FixedSizeChunker(chunk_size=200, chunk_overlap=40).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.doc_id == document.doc_id for chunk in chunks)
    assert all(chunk.char_end > chunk.char_start for chunk in chunks)
    starts = [chunk.char_start for chunk in chunks]
    assert starts == sorted(starts)


def test_overlap_actually_overlaps() -> None:
    """Without overlap a sentence spanning a boundary is retrievable from neither side.

    Run on unbroken text on purpose: overlap is applied when the splitter has to
    cut mid-run, which is exactly when a passage risks being severed.
    """
    document = make_document(text="word " * 200)

    overlapping = FixedSizeChunker(chunk_size=200, chunk_overlap=80).chunk(document)
    adjacent = FixedSizeChunker(chunk_size=200, chunk_overlap=0).chunk(document)

    assert all(b.char_start < a.char_end for a, b in pairwise(overlapping))
    assert all(b.char_start >= a.char_end for a, b in pairwise(adjacent))


def test_overlap_does_not_apply_where_a_natural_boundary_already_fits() -> None:
    """A property of the recursive splitter that shapes how phase 14 reads.

    Paragraphs shorter than the chunk size become chunks on their own, and
    overlap never engages — so a sweep over overlap values will barely move on a
    corpus of short paragraphs, and that is the splitter behaving correctly
    rather than the parameter failing to matter.
    """
    document = make_document(text="Alpha para.\n\nBeta para.\n\nGamma para.")

    without = FixedSizeChunker(chunk_size=200, chunk_overlap=0).chunk(document)
    generous = FixedSizeChunker(chunk_size=200, chunk_overlap=80).chunk(document)

    assert [(c.char_start, c.char_end) for c in without] == [
        (c.char_start, c.char_end) for c in generous
    ]


def test_short_document_becomes_one_chunk() -> None:
    document = make_document(text="Short enough to fit whole.")

    chunks = FixedSizeChunker(chunk_size=1000, chunk_overlap=200).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].text == "Short enough to fit whole."
    assert (chunks[0].char_start, chunks[0].char_end) == (0, len(document.text))


# --- determinism, which is the point ----------------------------------------


def test_chunk_ids_are_stable_across_runs() -> None:
    document = make_document()

    first = FixedSizeChunker(chunk_size=200, chunk_overlap=40).chunk(document)
    second = FixedSizeChunker(chunk_size=200, chunk_overlap=40).chunk(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]


def test_chunk_ids_are_stable_across_a_reload_of_the_same_document() -> None:
    """Re-ingesting an unchanged file must update in place, not duplicate."""
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=40)

    before = chunker.chunk(make_document())
    after = chunker.chunk(make_document())

    assert [chunk.chunk_id for chunk in before] == [chunk.chunk_id for chunk in after]


def test_chunk_ids_are_unique_within_a_document() -> None:
    chunks = FixedSizeChunker(chunk_size=120, chunk_overlap=20).chunk(make_document())

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


@pytest.mark.parametrize(
    ("size_a", "overlap_a", "size_b", "overlap_b"),
    [(200, 40, 400, 40), (200, 40, 200, 80)],
)
def test_different_configurations_produce_different_ids(
    size_a: int, overlap_a: int, size_b: int, overlap_b: int
) -> None:
    """The phase 14 sweep depends on this.

    Ids derived from position alone would let chunk 3 of one configuration
    collide with chunk 3 of another, and the phase 6 store upserts by chunk id —
    so one sweep would quietly overwrite the other.
    """
    document = make_document()

    a = FixedSizeChunker(chunk_size=size_a, chunk_overlap=overlap_a).chunk(document)
    b = FixedSizeChunker(chunk_size=size_b, chunk_overlap=overlap_b).chunk(document)

    assert {chunk.chunk_id for chunk in a}.isdisjoint({chunk.chunk_id for chunk in b})


def test_different_documents_produce_different_ids() -> None:
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=40)

    a = chunker.chunk(make_document(name="a.md"))
    b = chunker.chunk(make_document(name="b.md"))

    assert {chunk.chunk_id for chunk in a}.isdisjoint({chunk.chunk_id for chunk in b})


def test_chunk_id_is_scannable() -> None:
    """Phase 11 is a human reading these; the document and position stay legible."""
    identifier = chunk_id(
        fingerprint="fixed-1000-200", doc_id="0123456789abcdef", index=7, char_start=10, char_end=99
    )

    assert identifier.startswith("0123456789abcdef-0007-")


# --- configuration ----------------------------------------------------------


def test_chunker_defaults_come_from_settings() -> None:
    from neuronest.config import settings

    chunker = FixedSizeChunker()

    assert chunker.chunk_size == settings.chunk_size
    assert chunker.chunk_overlap == settings.chunk_overlap


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 200)],
)
def test_invalid_configurations_are_rejected(chunk_size: int, chunk_overlap: int) -> None:
    """A sweep constructs chunkers directly, bypassing the settings validator."""
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# --- many documents ---------------------------------------------------------


def test_chunk_documents_preserves_document_order() -> None:
    documents = [make_document(name=f"{letter}.md") for letter in ("a", "b", "c")]

    chunks = chunk_documents(documents, FixedSizeChunker(chunk_size=200, chunk_overlap=40))

    seen: list[str] = []
    for chunk in chunks:
        if chunk.doc_id not in seen:
            seen.append(chunk.doc_id)
    assert seen == [document.doc_id for document in documents]


def test_chunk_documents_over_an_empty_corpus() -> None:
    assert chunk_documents([], FixedSizeChunker()) == []


def test_chunks_are_frozen() -> None:
    chunk = FixedSizeChunker(chunk_size=200, chunk_overlap=40).chunk(make_document())[0]

    with pytest.raises(ValueError):
        chunk.text = "rewritten"  # type: ignore[misc]

