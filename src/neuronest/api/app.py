"""The FastAPI application.

Ingest, query, health and stats over the same core the CLI and the evaluation
harness use. No retrieval logic lives here: this layer parses requests, calls
into the pipeline and shapes the result, so the numbers phases 12-15 publish are
produced by exactly the code path a user hits.
"""

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, UploadFile

from neuronest.api.dependencies import Services
from neuronest.api.schemas import (
    HealthResponse,
    IngestFailure,
    IngestResponse,
    LocalIngestRequest,
    Passage,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from neuronest.config import settings
from neuronest.ingest.loaders import Document, LoadError, load_directory, load_document

DESCRIPTION = """
Retrieval-augmented generation that answers only from the passages it retrieved.

Ingest documents, ask a question, get the passages that support an answer — or
nothing at all, when the corpus does not contain one. An empty result is a
first-class outcome here, not an error.
"""


def create_app(services: Services | None = None) -> FastAPI:
    """Build the app.

    Takes its services rather than constructing them at import time, which is
    what lets the tests drive the whole HTTP surface with a stub embedder and a
    temporary index instead of half a gigabyte of model weights.
    """
    resolved = services if services is not None else Services()

    app = FastAPI(
        title="NeuroNest",
        description=DESCRIPTION,
        version="0.1.0",
        summary="Grounded question answering over your own documents.",
    )

    def get_services() -> Services:
        return resolved

    Injected = Annotated[Services, Depends(get_services)]

    def _ingest_documents(services: Services, documents: list[Document]) -> tuple[int, int]:
        chunks_written = 0
        for document in documents:
            chunks = services.chunker.chunk(document)
            services.store.ingest(document, chunks)
            chunks_written += len(chunks)
        return len(documents), chunks_written

    def _as_failure(error: LoadError) -> IngestFailure:
        return IngestFailure(
            source=str(error.source_path), code=error.code, message=error.message
        )

    @app.post("/ingest", response_model=IngestResponse, tags=["ingest"])
    async def ingest_upload(
        services: Injected, file: Annotated[UploadFile, File()]
    ) -> IngestResponse:
        """Ingest one uploaded document.

        The upload is written to a temporary file because the loaders read from
        disk — a PDF is parsed by path, not from a stream, and buffering it in
        memory to hand it a path would only add a copy.
        """
        name = Path(file.filename or "upload").name
        with tempfile.TemporaryDirectory() as workspace:
            staged = Path(workspace) / name
            staged.write_bytes(await file.read())

            # Identity comes from the uploaded name, not the temporary path, so
            # uploading the same file twice updates rather than duplicates.
            result = load_document(staged, identity=name)
            if isinstance(result, LoadError):
                return IngestResponse(
                    documents_ingested=0, chunks_written=0, failures=[_as_failure(result)]
                )

            documents, chunks = _ingest_documents(services, [result])
            return IngestResponse(documents_ingested=documents, chunks_written=chunks)

    @app.post("/ingest/local", response_model=IngestResponse, tags=["ingest"])
    async def ingest_local(services: Injected, request: LocalIngestRequest) -> IngestResponse:
        """Ingest a file or directory already on the server's filesystem.

        How a corpus is loaded in bulk, and how the evaluation harness in phases
        12-15 builds an index without pushing forty files through HTTP.
        """
        path = Path(request.path).expanduser()

        if path.is_dir():
            report = load_directory(path)
            documents, chunks = _ingest_documents(services, report.documents)
            return IngestResponse(
                documents_ingested=documents,
                chunks_written=chunks,
                failures=[_as_failure(error) for error in report.errors],
            )

        result = load_document(path)
        if isinstance(result, LoadError):
            return IngestResponse(
                documents_ingested=0, chunks_written=0, failures=[_as_failure(result)]
            )

        documents, chunks = _ingest_documents(services, [result])
        return IngestResponse(documents_ingested=documents, chunks_written=chunks)

    @app.post("/query", response_model=QueryResponse, tags=["query"])
    async def query(services: Injected, request: QueryRequest) -> QueryResponse:
        """Answer from the retrieved passages, or say why there is no answer.

        An empty ``passages`` list with ``refused: true`` is a successful
        response, not a 404. The corpus genuinely does not cover the question,
        and saying so is the behaviour this service exists to demonstrate.

        So is an answerless response with a ``note``: a provider outage still
        returns the passages, because they are the evidence and they are still
        correct.
        """
        retriever = services.retriever
        results = retriever.retrieve(request.question, k=request.k)

        answer: str | None = None
        note: str | None = None
        refused = not results

        if request.generate:
            generated = services.generator.generate(request.question, results)
            answer = generated.answer
            note = generated.note
            refused = generated.refused

        return QueryResponse(
            question=request.question,
            answer=answer,
            note=note,
            passages=[
                Passage(
                    chunk_id=item.chunk_id,
                    doc_id=item.doc_id,
                    text=item.text,
                    source_path=item.source_path,
                    char_start=item.char_start,
                    char_end=item.char_end,
                    score=item.score,
                    rank=item.rank,
                )
                for item in results
            ],
            refused=refused,
            score_threshold=retriever.score_threshold,
        )

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health(services: Injected) -> HealthResponse:
        """Report state that was actually checked.

        Opens the collection and counts it, and asks the embedder for the width
        it really produces — which loads the model if it is not loaded yet. Both
        can fail, and when they do this says so and reports unhealthy rather
        than returning ok because the process happens to be answering.
        """
        store = services.store
        model_name = services.embedder.model_name

        try:
            chunks = store.count()
            dimensions = services.embedder.dimensions
        except Exception as exc:
            return HealthResponse(
                status="unhealthy",
                store_reachable=False,
                embedding_model=model_name,
                embedding_dimensions=None,
                collection=store.collection_name,
                chunks_indexed=None,
                detail=f"{type(exc).__name__}: {exc}",
            )

        return HealthResponse(
            status="ok",
            store_reachable=True,
            embedding_model=model_name,
            embedding_dimensions=dimensions,
            collection=store.collection_name,
            chunks_indexed=chunks,
        )

    @app.get("/stats", response_model=StatsResponse, tags=["operations"])
    async def stats(services: Injected) -> StatsResponse:
        """Corpus size and the configuration that produced it."""
        store = services.store
        chunker = services.chunker

        chunk_size = getattr(chunker, "chunk_size", settings.chunk_size)
        chunk_overlap = getattr(chunker, "chunk_overlap", settings.chunk_overlap)

        return StatsResponse(
            documents=store.document_count(),
            chunks=store.count(),
            embedding_model=services.embedder.model_name,
            embedding_dimensions=services.embedder.dimensions,
            collection=store.collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=services.retriever.top_k,
            score_threshold=services.retriever.score_threshold,
            generation_model=services.generator.model_name,
            generation_available=services.generator.available,
        )

    @app.delete("/corpus", status_code=204, tags=["operations"])
    async def reset_corpus(services: Injected) -> None:
        """Drop everything indexed.

        Exists because phases 12-15 rebuild the index between sweeps, and doing
        it through the API keeps the harness on the same path a user takes.
        """
        services.store.reset()

    return app


app = create_app()
