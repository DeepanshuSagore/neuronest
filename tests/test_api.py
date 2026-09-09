"""The HTTP surface, driven end to end against a stub embedder.

The app takes its services as an argument precisely so this file can exist: a
temporary index and a keyword-counting embedder let every endpoint be exercised
offline in milliseconds. Tested against real weights, these would download a
model and then get marked slow and skipped, which is how an API ends up with no
coverage of its own contract.
"""

import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neuronest.api.app import create_app
from neuronest.api.dependencies import Services
from neuronest.embed.base import Vector
from neuronest.ingest.chunking import FixedSizeChunker

KEYWORDS = ("alpha", "beta", "gamma")


class KeywordEmbedder:
    """Counts three keywords and normalises, so relevance is predictable."""

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
            raw = [float(lowered.count(word)) for word in KEYWORDS]
            magnitude = math.sqrt(sum(value * value for value in raw)) or 1.0
            vectors.append([value / magnitude for value in raw])
        return vectors


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    services = Services(
        embedder=KeywordEmbedder(),
        chunker=FixedSizeChunker(chunk_size=200, chunk_overlap=0),
        chroma_path=tmp_path / "chroma",
    )
    with TestClient(create_app(services)) as test_client:
        yield test_client


def upload(client: TestClient, name: str, body: str) -> dict[str, object]:
    response = client.post("/ingest", files={"file": (name, body.encode("utf-8"), "text/markdown")})
    assert response.status_code == 200, response.text
    return dict(response.json())


# --- ingest -----------------------------------------------------------------


def test_uploading_a_document_indexes_it(client: TestClient) -> None:
    payload = upload(client, "notes.md", "alpha beta gamma")

    assert payload["documents_ingested"] == 1
    assert payload["chunks_written"] == 1
    assert payload["failures"] == []


def test_uploading_the_same_name_twice_does_not_duplicate(client: TestClient) -> None:
    """Identity comes from the uploaded name, not the temporary path it was staged at."""
    upload(client, "notes.md", "alpha beta gamma")
    upload(client, "notes.md", "alpha beta gamma")

    assert client.get("/stats").json()["chunks"] == 1


def test_a_rejected_upload_reports_why(client: TestClient) -> None:
    """A bad file is a described outcome, not a 500."""
    payload = upload(client, "empty.md", "")

    assert payload["documents_ingested"] == 0
    failures = payload["failures"]
    assert isinstance(failures, list)
    assert failures[0]["code"] == "empty_file"


def test_an_unsupported_type_reports_why(client: TestClient) -> None:
    payload = upload(client, "sheet.csv", "a,b\n1,2\n")

    assert payload["failures"][0]["code"] == "unsupported_type"  # type: ignore[index]


def test_ingesting_a_local_directory(client: TestClient, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("alpha alpha", encoding="utf-8")
    (corpus / "b.md").write_text("beta beta", encoding="utf-8")
    (corpus / "broken.md").write_text("   ", encoding="utf-8")

    response = client.post("/ingest/local", json={"path": str(corpus)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["documents_ingested"] == 2
    assert [failure["code"] for failure in payload["failures"]] == ["empty_file"]


def test_ingesting_a_missing_local_path_reports_why(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/ingest/local", json={"path": str(tmp_path / "absent.md")})

    assert response.status_code == 200
    assert response.json()["failures"][0]["code"] == "not_found"


# --- query ------------------------------------------------------------------


def test_querying_returns_ranked_passages(client: TestClient) -> None:
    upload(client, "alpha.md", "alpha alpha alpha")
    upload(client, "beta.md", "beta beta beta")

    payload = client.post("/query", json={"question": "alpha"}).json()

    assert payload["refused"] is False
    assert payload["passages"][0]["text"] == "alpha alpha alpha"
    assert payload["passages"][0]["rank"] == 1
    assert payload["passages"][0]["score"] > 0.9
    assert payload["passages"][0]["source_path"].endswith("alpha.md")


def test_querying_an_absent_topic_refuses(client: TestClient) -> None:
    """The behaviour the service exists to demonstrate: empty is a success, not a 404."""
    upload(client, "alpha.md", "alpha alpha alpha")

    response = client.post("/query", json={"question": "gamma"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["passages"] == []
    assert payload["refused"] is True


def test_querying_an_empty_corpus_refuses(client: TestClient) -> None:
    payload = client.post("/query", json={"question": "alpha"}).json()

    assert payload["refused"] is True


def test_query_honours_k(client: TestClient) -> None:
    upload(client, "a.md", "alpha alpha")
    upload(client, "b.md", "alpha beta")

    payload = client.post("/query", json={"question": "alpha", "k": 1}).json()

    assert len(payload["passages"]) == 1


def test_query_reports_the_threshold_it_applied(client: TestClient) -> None:
    payload = client.post("/query", json={"question": "alpha"}).json()

    assert isinstance(payload["score_threshold"], float)


@pytest.mark.parametrize("body", [{"question": ""}, {}, {"question": "alpha", "k": 0}])
def test_invalid_queries_are_rejected(client: TestClient, body: dict[str, object]) -> None:
    assert client.post("/query", json=body).status_code == 422


# --- stats ------------------------------------------------------------------


def test_stats_counts_documents_and_chunks(client: TestClient) -> None:
    upload(client, "a.md", "alpha alpha")
    upload(client, "b.md", "beta beta")

    payload = client.get("/stats").json()

    assert payload["documents"] == 2
    assert payload["chunks"] == 2
    assert payload["embedding_model"] == "keyword-v1"
    assert payload["embedding_dimensions"] == 3


def test_stats_reports_the_active_configuration(client: TestClient) -> None:
    """Every published number has to trace to the configuration that produced it."""
    payload = client.get("/stats").json()

    assert payload["chunk_size"] == 200
    assert payload["chunk_overlap"] == 0
    assert isinstance(payload["top_k"], int)
    assert isinstance(payload["score_threshold"], float)


def test_stats_flags_metrics_as_placeholder(client: TestClient) -> None:
    assert client.get("/stats").json()["metrics_are_placeholder"] is True


# --- corpus reset -----------------------------------------------------------


def test_resetting_the_corpus_empties_it(client: TestClient) -> None:
    upload(client, "a.md", "alpha alpha")
    assert client.get("/stats").json()["chunks"] == 1

    assert client.delete("/corpus").status_code == 204

    assert client.get("/stats").json()["chunks"] == 0


# --- the documented contract ------------------------------------------------


def test_openapi_describes_every_endpoint(client: TestClient) -> None:
    """/docs renders from this, so a missing path here is a missing page there."""
    paths = client.get("/openapi.json").json()["paths"]

    assert {"/ingest", "/ingest/local", "/query", "/stats", "/corpus"} <= set(paths)


def test_docs_page_renders(client: TestClient) -> None:
    response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger" in response.text.lower()
