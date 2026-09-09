"""A content-addressed embedding cache, and the wrapper that uses it.

This is not a performance nicety. Phases 12 to 15 re-embed the same corpus
across chunking sweeps and model comparisons; without a cache every re-run pays
the full cost again, and the experiments stop being something you re-run
casually. Making them cheap is what keeps the numbers honest, because a
measurement you are reluctant to repeat is one you stop checking.

Keyed on ``(model_name, sha256(text))``. Content-addressed rather than keyed by
chunk id on purpose: two chunking configurations that happen to produce the
same passage share the cached vector, which is exactly the overlap a chunk-size
sweep generates.
"""

import hashlib
import sqlite3
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from neuronest.embed.base import Embedder, Vector

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    model      TEXT    NOT NULL,
    text_hash  TEXT    NOT NULL,
    dimensions INTEGER NOT NULL,
    vector     BLOB    NOT NULL,
    PRIMARY KEY (model, text_hash)
) WITHOUT ROWID;
"""


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheStats:
    """Hits and misses since the wrapper was constructed.

    Phase 5 exits on reporting a hit rate, and phases 12-15 use it to tell a
    genuinely cheap re-run from one that quietly re-embedded everything.
    """

    hits: int
    misses: int

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class EmbeddingCache:
    """SQLite-backed vector storage.

    SQLite rather than a directory of files: a sweep produces tens of thousands
    of vectors, and one file each turns the cache into an inode problem and
    makes it slow to copy or delete.

    Vectors are stored as float32. The models here emit float32 anyway, so this
    round-trips exactly while halving the bytes against Python's native float.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(_SCHEMA)
        self._connection.commit()

    def get_many(self, model: str, hashes: Sequence[str]) -> dict[str, Vector]:
        """Fetch every cached vector for these hashes in one round trip."""
        if not hashes:
            return {}

        found: dict[str, Vector] = {}
        # SQLite caps host parameters per statement, so a corpus-sized lookup is
        # chunked rather than sent as one enormous IN clause.
        for offset in range(0, len(hashes), 500):
            window = hashes[offset : offset + 500]
            placeholders = ",".join("?" * len(window))
            rows = self._connection.execute(
                f"SELECT text_hash, vector FROM embeddings WHERE model = ? AND text_hash IN ({placeholders})",  # noqa: E501
                (model, *window),
            ).fetchall()
            for stored_hash, blob in rows:
                vector = array("f")
                vector.frombytes(blob)
                found[stored_hash] = vector.tolist()

        return found

    def put_many(self, model: str, vectors: dict[str, Vector]) -> None:
        if not vectors:
            return

        self._connection.executemany(
            "INSERT OR REPLACE INTO embeddings (model, text_hash, dimensions, vector) "
            "VALUES (?, ?, ?, ?)",
            [
                (model, stored_hash, len(vector), array("f", vector).tobytes())
                for stored_hash, vector in vectors.items()
            ],
        )
        self._connection.commit()

    def count(self, model: str | None = None) -> int:
        if model is None:
            row = self._connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class CachingEmbedder:
    """An embedder that consults the cache before calling the one it wraps.

    Satisfies ``Embedder`` itself, so callers cannot tell a cached embedder from
    a bare one and nothing downstream has to know the cache exists.
    """

    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache
        self._hits = 0
        self._misses = 0

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def stats(self) -> CacheStats:
        return CacheStats(hits=self._hits, misses=self._misses)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []

        hashes = [text_hash(text) for text in texts]
        cached = self._cache.get_many(self.model_name, hashes)

        # Deduplicated before anything is computed. A chunk-size sweep produces
        # the same passage under several configurations, and a corpus repeats
        # boilerplate; embedding each distinct text once is most of what makes
        # a cold run cheap, before the cache has helped at all.
        missing: dict[str, str] = {}
        for text, digest in zip(texts, hashes, strict=True):
            if digest not in cached and digest not in missing:
                missing[digest] = text

        self._hits += sum(1 for digest in hashes if digest in cached)
        self._misses += sum(1 for digest in hashes if digest not in cached)

        if missing:
            computed = self._inner.embed(list(missing.values()))
            fresh = dict(zip(missing.keys(), computed, strict=True))
            self._cache.put_many(self.model_name, fresh)
            cached.update(fresh)

        return [cached[digest] for digest in hashes]
