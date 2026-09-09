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
from typing import Protocol, runtime_checkable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict

from neuronest.config import settings
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

        self._splitter = RecursiveCharacterTextSplitter(
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


def chunk_documents(documents: list[Document], chunker: Chunker) -> list[Chunk]:
    """Chunk many documents, preserving the order they were given in."""
    return [chunk for document in documents for chunk in chunker.chunk(document)]
