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
    # Retrieval-only is an explicit mode rather than an accident of a missing
    # key, because the evaluation harness needs it. Phases 12 and 14-15 sweep
    # retrieval over the whole question set many times, and generating an answer
    # on every one of those queries would put a bill on a measurement that is
    # supposed to be free to repeat.
    generate: bool = Field(
        default=True, description="Set false to retrieve passages without generating an answer."
    )


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
    #: True when the service declined to answer. Either nothing cleared the
    #: relevance threshold, in which case the model was never called, or the
    #: model read the passages and reported that they do not answer the
    #: question. An empty ``passages`` list tells the two apart.
    refused: bool
    score_threshold: float
    #: The answer, grounded in the passages below it. Bracketed numbers in the
    #: text are passage ranks: ``[2]`` is the passage reported at rank 2.
    answer: str | None = None
    #: Why there is no answer, when there is no answer. Set on a refusal, on a
    #: provider failure, and when no API key is configured — the passages are
    #: returned in all three cases, because they are still evidence.
    note: str | None = None


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
    generation_model: str
    #: Whether an API key is configured. Answers the question a user asks after
    #: getting passages and a note instead of an answer.
    generation_available: bool
    metrics_are_placeholder: bool = True
