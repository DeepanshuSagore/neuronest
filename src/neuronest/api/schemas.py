"""Request and response shapes.

Declared as pydantic models rather than dicts so the OpenAPI page at /docs is
generated from the same definitions the endpoints validate against, and cannot
describe a response the service does not actually return.
"""

from pydantic import BaseModel, Field

from neuronest.ingest.loaders import LoadErrorCode


class IngestFailure(BaseModel):
    """One file that did not make it in, and why."""

    source: str
    code: LoadErrorCode
    message: str


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_written: int
    failures: list[IngestFailure] = Field(default_factory=list)


class LocalIngestRequest(BaseModel):
    path: str = Field(description="A file or directory on the server's filesystem.")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int | None = Field(default=None, gt=0, description="Defaults to the configured top_k.")


class Passage(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    char_start: int
    char_end: int
    score: float
    rank: int


class QueryResponse(BaseModel):
    question: str
    passages: list[Passage]
    #: True when nothing cleared the relevance threshold. Phase 10 refuses on
    #: this without calling the model, so it is stated rather than left to be
    #: inferred from an empty list.
    refused: bool
    score_threshold: float


class HealthResponse(BaseModel):
    """Real state, checked on every call.

    A hard-coded ``{"status": "ok"}`` proves only that the process is running,
    which is the one thing you already knew because it answered. Each field
    below is read from the thing it describes.
    """

    status: str
    store_reachable: bool
    embedding_model: str
    embedding_dimensions: int | None
    collection: str
    chunks_indexed: int | None
    detail: str | None = None


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    embedding_model: str
    embedding_dimensions: int
    collection: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    metrics_are_placeholder: bool = True
