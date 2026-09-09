"""Embedder and cache behaviour.

Almost everything here runs against a stub rather than a real model. The cache
is the part with logic worth testing — deduplication, ordering, scoping,
persistence — and none of it needs 500MB of torch to exercise. The two tests
that do need real weights or a live key are marked ``slow`` and excluded from a
default run, so the suite stays offline and secret-free.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from neuronest.embed.base import Embedder, Vector
from neuronest.embed.cache import CachingEmbedder, EmbeddingCache, text_hash


class StubEmbedder:
    """Deterministic, and records exactly what it was asked to embed.

    Vector components are small integers so they survive the cache's float32
    storage exactly; the precision tradeoff gets its own test rather than
    contaminating every other assertion.
    """

    def __init__(self, name: str = "stub-v1", dimensions: int = 4) -> None:
        self._name = name
        self._dimensions = dimensions
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def vector_for(self, text: str) -> Vector:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        return [float((seed >> (8 * index)) & 0xFF) for index in range(self._dimensions)]

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.calls.append(list(texts))
        return [self.vector_for(text) for text in texts]

    @property
    def texts_embedded(self) -> int:
        return sum(len(call) for call in self.calls)


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "embeddings.db"


# --- the protocol -----------------------------------------------------------


def test_stub_satisfies_the_protocol() -> None:
    assert isinstance(StubEmbedder(), Embedder)


def test_caching_embedder_satisfies_the_protocol(cache_path: Path) -> None:
    """Callers must not be able to tell a cached embedder from a bare one."""
    with EmbeddingCache(cache_path) as cache:
        assert isinstance(CachingEmbedder(StubEmbedder(), cache), Embedder)


def test_caching_embedder_reports_the_wrapped_model(cache_path: Path) -> None:
    with EmbeddingCache(cache_path) as cache:
        wrapped = CachingEmbedder(StubEmbedder(name="all-MiniLM-L6-v2", dimensions=8), cache)

        assert wrapped.model_name == "all-MiniLM-L6-v2"
        assert wrapped.dimensions == 8


# --- the exit criterion: embed twice, second run hits the cache -------------


def test_first_run_misses_and_second_run_hits(cache_path: Path) -> None:
    texts = ["alpha", "beta", "gamma"]
    inner = StubEmbedder()

    with EmbeddingCache(cache_path) as cache:
        embedder = CachingEmbedder(inner, cache)

        first = embedder.embed(texts)
        assert embedder.stats.hits == 0
        assert embedder.stats.misses == 3
        assert inner.texts_embedded == 3

        second = embedder.embed(texts)

    assert second == first
    assert embedder.stats.hits == 3
    assert embedder.stats.hit_rate == pytest.approx(0.5)
    # The second pass computed nothing at all.
    assert inner.texts_embedded == 3


def test_hit_rate_is_zero_before_any_lookup(cache_path: Path) -> None:
    with EmbeddingCache(cache_path) as cache:
        assert CachingEmbedder(StubEmbedder(), cache).stats.hit_rate == 0.0


def test_cache_survives_being_reopened(cache_path: Path) -> None:
    """The cache is on disk, so a fresh process must not pay for the corpus again."""
    texts = ["alpha", "beta"]

    with EmbeddingCache(cache_path) as cache:
        CachingEmbedder(StubEmbedder(), cache).embed(texts)

    reopened_inner = StubEmbedder()
    with EmbeddingCache(cache_path) as cache:
        embedder = CachingEmbedder(reopened_inner, cache)
        embedder.embed(texts)

    assert embedder.stats.misses == 0
    assert reopened_inner.calls == []


# --- deduplication and ordering ---------------------------------------------


def test_repeated_text_within_one_batch_is_embedded_once(cache_path: Path) -> None:
    """A chunk-size sweep repeats passages; a corpus repeats boilerplate."""
    inner = StubEmbedder()

    with EmbeddingCache(cache_path) as cache:
        vectors = CachingEmbedder(inner, cache).embed(["same", "same", "same", "other"])

    assert inner.texts_embedded == 2
    assert vectors[0] == vectors[1] == vectors[2]
    assert vectors[3] != vectors[0]


def test_order_is_preserved_when_hits_and_misses_are_mixed(cache_path: Path) -> None:
    inner = StubEmbedder()

    with EmbeddingCache(cache_path) as cache:
        embedder = CachingEmbedder(inner, cache)
        embedder.embed(["beta"])

        vectors = embedder.embed(["alpha", "beta", "gamma"])

    assert vectors == [inner.vector_for(t) for t in ("alpha", "beta", "gamma")]


def test_empty_input_touches_nothing(cache_path: Path) -> None:
    inner = StubEmbedder()

    with EmbeddingCache(cache_path) as cache:
        assert CachingEmbedder(inner, cache).embed([]) == []

    assert inner.calls == []


# --- correctness of what is stored ------------------------------------------


def test_cached_vectors_match_freshly_computed_ones(cache_path: Path) -> None:
    inner = StubEmbedder()

    with EmbeddingCache(cache_path) as cache:
        embedder = CachingEmbedder(inner, cache)
        fresh = embedder.embed(["alpha", "beta"])
        from_cache = embedder.embed(["alpha", "beta"])

    assert from_cache == fresh == [inner.vector_for("alpha"), inner.vector_for("beta")]


def test_storage_is_float32(cache_path: Path) -> None:
    """A documented tradeoff, asserted rather than assumed.

    float32 halves the cache against Python's native float and is what the
    models emit anyway, but it does round values that a float64 would keep.
    """

    class PreciseEmbedder(StubEmbedder):
        def embed(self, texts: Sequence[str]) -> list[Vector]:
            self.calls.append(list(texts))
            return [[0.1234567890123456, 1.0, 2.0, 3.0] for _ in texts]

    with EmbeddingCache(cache_path) as cache:
        embedder = CachingEmbedder(PreciseEmbedder(), cache)
        fresh = embedder.embed(["x"])[0]
        stored = embedder.embed(["x"])[0]

    assert stored != fresh
    assert stored == pytest.approx(fresh, abs=1e-7)


def test_cache_is_scoped_per_model(cache_path: Path) -> None:
    """Phase 15 runs several models over one corpus through the same cache file."""
    mini = StubEmbedder(name="mini", dimensions=4)
    large = StubEmbedder(name="large", dimensions=4)

    with EmbeddingCache(cache_path) as cache:
        CachingEmbedder(mini, cache).embed(["alpha"])
        large_embedder = CachingEmbedder(large, cache)
        large_embedder.embed(["alpha"])

        assert cache.count("mini") == 1
        assert cache.count("large") == 1
        assert cache.count() == 2

    assert large_embedder.stats.misses == 1


def test_text_hash_is_content_addressed() -> None:
    assert text_hash("alpha") == text_hash("alpha")
    assert text_hash("alpha") != text_hash("beta")


# --- the hosted embedder ----------------------------------------------------


def test_mistral_embedder_refuses_to_construct_without_a_key() -> None:
    """Opt-in, and it says so rather than failing later with a 401."""
    from neuronest.embed.mistral import MistralEmbedder

    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        MistralEmbedder(api_key="")


# --- the real thing ---------------------------------------------------------


@pytest.mark.slow
def test_local_embedder_produces_normalised_vectors_of_the_expected_size() -> None:
    """Needs the weights on disk. Run with: uv run pytest -m slow"""
    from neuronest.embed.local import LocalEmbedder

    embedder = LocalEmbedder()
    vectors = embedder.embed(["retrieval is not generation", "refusal is the hard part"])

    assert len(vectors) == 2
    assert all(len(vector) == embedder.dimensions for vector in vectors)
    for vector in vectors:
        magnitude = sum(value * value for value in vector) ** 0.5
        assert magnitude == pytest.approx(1.0, abs=1e-5)


@pytest.mark.slow
def test_local_embedder_is_deterministic() -> None:
    """Same text, same vector — or the cache would be returning stale answers."""
    from neuronest.embed.local import LocalEmbedder

    embedder = LocalEmbedder()

    assert embedder.embed(["stability"]) == embedder.embed(["stability"])
