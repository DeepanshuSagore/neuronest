"""Splitting documents into the passages that actually get embedded and retrieved.

Written against a protocol rather than a single class because phase 14 adds a
semantic strategy and measures it against this one. Defining the interface now
costs a few lines; retrofitting one around a concrete class after four other
modules import it does not.

Two properties matter more than the splitting itself:

*Determinism.* The same document under the same configuration must produce the
same chunk ids on every run, or the labelled evaluation set in phase 11 rots
the first time the corpus is re-ingested.

*Configuration is part of a chunk's identity.* Phase 14 sweeps chunk sizes and
overlaps. If ids were derived from position alone, chunk 3 of a 500-character
run and chunk 3 of a 1500-character run would collide on the same id with
different text, and the phase 6 store — which upserts by chunk id — would
silently overwrite one sweep with another. The chunker's fingerprint is folded
into every id so two configurations can coexist and be compared.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from neuronest.config import settings
from neuronest.embed.base import Embedder, Vector
from neuronest.ingest.loaders import Document


class Chunk(BaseModel):
    """One passage of a document, with its position in the original text.

    ``char_start`` and ``char_end`` index into ``Document.text``, so a citation
    can point at the source rather than at a copy of it. Frozen for the same
    reason ``Document`` is: embeddings and store keys are derived from these
    fields.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int


@runtime_checkable
class Chunker(Protocol):
    """A strategy for splitting a document into chunks.

    ``fingerprint`` identifies the strategy *and* its parameters. It ends up
    inside every chunk id and in the phase 12 result files, which is what makes
    a published number traceable back to the configuration that produced it.
    """

    @property
    def fingerprint(self) -> str: ...

    def chunk(self, document: Document) -> list[Chunk]: ...


def chunk_id(*, fingerprint: str, doc_id: str, index: int, char_start: int, char_end: int) -> str:
    """Build a chunk id that is stable, unique per configuration, and scannable.

    The leading document id and ordinal are there for phase 11: a human
    labelling a hundred questions by chunk id needs to see which document and
    roughly where without decoding a hash. The digest behind them is what keeps
    two chunking configurations from colliding.
    """
    key = f"{fingerprint}|{doc_id}|{char_start}|{char_end}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}-{index:04d}-{digest}"


class FixedSizeChunker:
    """Fixed-size chunking over LangChain's ``RecursiveCharacterTextSplitter``.

    The splitter backs off through paragraph, line, space and finally raw
    character boundaries, so a chunk breaks at the most natural place that still
    fits the size — which is why it beats a naive slice at the same size.

    Sizes are in characters, not tokens, matching the settings they default to.
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

        # Settings validate this pair at startup, but a sweep in phase 14
        # constructs chunkers directly and bypasses that check entirely.
        if self.chunk_size <= 0:
            msg = f"chunk_size must be positive, got {self.chunk_size}"
            raise ValueError(msg)
        if self.chunk_overlap < 0:
            msg = f"chunk_overlap cannot be negative, got {self.chunk_overlap}"
            raise ValueError(msg)
        if self.chunk_overlap >= self.chunk_size:
            msg = (
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
            raise ValueError(msg)

        # Imported here rather than at module scope: langchain-core costs about
        # four seconds to import, and paying that to read a config value, serve
        # /health or run the CLI would be four seconds of cold start for nothing.
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # The offsets are read back off the splitter rather than recovered
            # with str.find afterwards. Overlapping chunks repeat text, so a
            # search for the second occurrence of a repeated passage finds the
            # first one and reports a position that is quietly wrong.
            add_start_index=True,
        )

    @property
    def fingerprint(self) -> str:
        return f"fixed-{self.chunk_size}-{self.chunk_overlap}"

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []

        for index, split in enumerate(self._splitter.create_documents([document.text])):
            char_start = int(split.metadata["start_index"])
            char_end = char_start + len(split.page_content)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(
                        fingerprint=self.fingerprint,
                        doc_id=document.doc_id,
                        index=index,
                        char_start=char_start,
                        char_end=char_end,
                    ),
                    doc_id=document.doc_id,
                    text=split.page_content,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

        return chunks


# A unit is a paragraph or a sentence, whichever comes first. Two alternatives:
# a blank line, which is a structural break in fixed-width text and separates
# headings, tables and ABNF blocks from prose; or a sentence terminator followed
# by whitespace.
#
# The second lookbehind is the part that is not obvious. Without it, "3.  Method"
# splits into "3." and "Method", and RFC section numbering makes that pattern
# appear thousands of times across this corpus — every heading, every
# cross-reference. Refusing to split when a digit precedes the period costs the
# split after a sentence that genuinely ends in a number ("...defined in RFC
# 7231.  The following"), which is the rarer case by a wide margin here.
_UNIT_BOUNDARY = re.compile(r"\n[ \t]*\n\s*|(?<=[.!?])(?<![0-9][.!?])\s+")


def _units(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences and paragraphs in ``text``.

    Spans index into the original string and exclude the whitespace between
    them, so a chunk assembled from units can be sliced straight back out of the
    document and its offsets stay true.
    """
    spans: list[tuple[int, int]] = []
    position = 0

    def keep(start: int, end: int) -> None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))

    for boundary in _UNIT_BOUNDARY.finditer(text):
        keep(position, boundary.start())
        position = boundary.end()
    keep(position, len(text))

    return spans


def _cosine_distance(left: Vector, right: Vector) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return 1.0 - dot / (left_norm * right_norm)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Linear-interpolated percentile, matching numpy's default method.

    Implemented here rather than imported: numpy arrives transitively through
    torch, and depending on it from the chunker would make a module that the API
    imports at startup depend on the embedding stack for one arithmetic helper.
    """
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _merge_short(
    spans: list[tuple[int, int]], minimum: int, maximum: int
) -> list[tuple[int, int]]:
    """Fold a runt back into the span before it.

    ``min_chunk_chars`` suppresses a break mid-document, but it cannot help the
    last chunk: a document whose final sentence falls just after a boundary ends
    with a chunk a few characters long. Merging keeps the maximum, so a runt
    that cannot be absorbed is left alone rather than pushed past the embedder's
    window.
    """
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and end - start < minimum and end - merged[-1][0] <= maximum:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


class SemanticChunker:
    """Chunking that breaks where the subject changes rather than at a length.

    Sentences are embedded and compared with their neighbours; the widest gaps
    become chunk boundaries. The promise is that a chunk holds one idea instead
    of one thousand characters, so a passage retrieved for a question contains
    the whole answer rather than the half that fell on the right side of an
    arbitrary cut.

    Three parameters exist because the naive algorithm fails in three specific
    ways on real documents:

    ``buffer_size`` embeds each sentence together with its neighbours. A bare
    four-word sentence embeds to something close to noise, and noise reads as a
    large distance, which is a break in the wrong place.

    ``min_chunk_chars`` suppresses a break while the current chunk is still
    short. Headings sit far from everything around them in embedding space, so
    without this the corpus acquires hundreds of chunks that contain a section
    title and nothing else.

    ``max_chunk_chars`` forces one. The default embedding model truncates at 256
    word pieces — roughly a thousand characters — so text past that point in a
    chunk is stored but never actually embedded, and a chunk that is never fully
    embedded cannot be retrieved for what is written at the end of it.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        breakpoint_percentile: float = 95.0,
        buffer_size: int = 1,
        min_chunk_chars: int = 200,
        max_chunk_chars: int = 2000,
    ) -> None:
        if not 0.0 < breakpoint_percentile < 100.0:
            msg = f"breakpoint_percentile must be between 0 and 100, got {breakpoint_percentile}"
            raise ValueError(msg)
        if buffer_size < 0:
            msg = f"buffer_size cannot be negative, got {buffer_size}"
            raise ValueError(msg)
        if min_chunk_chars < 0:
            msg = f"min_chunk_chars cannot be negative, got {min_chunk_chars}"
            raise ValueError(msg)
        if max_chunk_chars <= min_chunk_chars:
            msg = (
                f"max_chunk_chars ({max_chunk_chars}) must be larger than "
                f"min_chunk_chars ({min_chunk_chars})"
            )
            raise ValueError(msg)

        self._embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    @property
    def fingerprint(self) -> str:
        # The embedder is deliberately absent. It does change the boundaries, so
        # two models under these parameters are not the same chunking — but the
        # model already names the collection and is stamped into every result
        # file, and folding it in here would put it in the chunk ids as well,
        # where it buys nothing: different boundaries already produce different
        # ids.
        percentile = f"{self.breakpoint_percentile:g}"
        return (
            f"semantic-{percentile}-{self.buffer_size}-"
            f"{self.min_chunk_chars}-{self.max_chunk_chars}"
        )

    def chunk(self, document: Document) -> list[Chunk]:
        units = _units(document.text)
        if not units:
            return []

        return [
            Chunk(
                chunk_id=chunk_id(
                    fingerprint=self.fingerprint,
                    doc_id=document.doc_id,
                    index=index,
                    char_start=char_start,
                    char_end=char_end,
                ),
                doc_id=document.doc_id,
                text=document.text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
            )
            for index, (char_start, char_end) in enumerate(self._spans(document.text, units))
        ]

    def _breakpoints(self, text: str, units: Sequence[tuple[int, int]]) -> set[int]:
        """Indices of the units that start a new chunk, on distance alone."""
        if len(units) < 2:
            return set()

        windows: list[str] = []
        for index in range(len(units)):
            first = max(0, index - self.buffer_size)
            last = min(len(units) - 1, index + self.buffer_size)
            windows.append(text[units[first][0] : units[last][1]])

        vectors = self._embedder.embed(windows)
        distances = [
            _cosine_distance(vectors[index], vectors[index + 1])
            for index in range(len(vectors) - 1)
        ]
        cut = _percentile(distances, self.breakpoint_percentile)
        return {index + 1 for index, distance in enumerate(distances) if distance > cut}

    def _spans(self, text: str, units: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        breaks = self._breakpoints(text, units)
        spans: list[tuple[int, int]] = []
        first = 0

        for index in range(1, len(units) + 1):
            start = units[first][0]
            end = units[index - 1][1]
            if index == len(units):
                spans.extend(self._capped(start, end))
                break

            grown = units[index][1] - start
            wants_break = index in breaks and end - start >= self.min_chunk_chars
            if wants_break or grown > self.max_chunk_chars:
                spans.extend(self._capped(start, end))
                first = index

        return _merge_short(spans, self.min_chunk_chars, self.max_chunk_chars)

    def _capped(self, start: int, end: int) -> list[tuple[int, int]]:
        """Split a span that no embedder would read to the end of.

        Sliced into equal pieces rather than repeated cuts at the maximum, which
        would leave a final sliver of whatever is left over.
        """
        length = end - start
        if length <= self.max_chunk_chars:
            return [(start, end)]

        pieces = math.ceil(length / self.max_chunk_chars)
        size = math.ceil(length / pieces)
        return [(at, min(at + size, end)) for at in range(start, end, size)]


def chunk_documents(documents: list[Document], chunker: Chunker) -> list[Chunk]:
    """Chunk many documents, preserving the order they were given in."""
    return [chunk for document in documents for chunk in chunker.chunk(document)]
