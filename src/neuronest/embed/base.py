"""The interface every embedder implements.

Phase 15 benchmarks three models against each other, so the thing that varies
has to sit behind one interface from the start. ``model_name`` is part of that
interface rather than an implementation detail: it keys the cache and it is
stamped into every evaluation result file, which is what makes a published
number traceable to the model that produced it.

Vectors are plain lists of floats rather than numpy arrays. Chroma accepts
lists, the hosted embedder returns lists, and keeping numpy out of the boundary
means the protocol does not depend on the local implementation's stack.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

Vector = list[float]


@runtime_checkable
class Embedder(Protocol):
    @property
    def model_name(self) -> str:
        """Identifies the model *and* anything that changes its output.

        Two embedders that could produce different vectors must never report
        the same name, because the cache trusts this to decide whether a stored
        vector is still valid.
        """
        ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed texts, returning one vector per input in the same order.

        Queries and passages go through this same method. None of the models
        this project compares use an asymmetric query prefix; one that did
        would need the protocol widened rather than a prefix smuggled in here.
        """
        ...
