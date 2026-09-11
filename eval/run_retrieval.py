"""Recall@k and MRR over the hand-labelled evaluation set.

Retrieval is scored here on its own, with no language model anywhere in the
path. That separation is the reason this project exists: most RAG failures are
retrieval failures that get blamed on the model, and the only way to tell those
two apart is to measure the half that can be measured without the other.

Two definitions decide every number this prints.

*A passage is relevant if it overlaps a labelled span.* Not "if its chunk id is
the labelled one". Chunk ids fold in the chunker's fingerprint, so scoring by id
would make phase 14's sweep unscoreable the moment it changed the chunk size.
The labels are character ranges in the source documents; overlap is the test,
and it holds under any chunking.

*Coverage rank is the rank at which every labelled span has been seen.* For a
question with one label that is the ordinary first-relevant rank. For a
multi-chunk question it is the rank at which the question first becomes fully
supported, which is what "the answer needs two chunks" actually asks of
retrieval. Recall@k is then ``coverage_rank <= k`` and MRR is
``1 / coverage_rank``, both falling out of one quantity rather than two
definitions that disagree at the edges.

The score threshold is deliberately off while measuring. Filtering at 0.25 would
fold the refusal policy into the recall number, and keeping those apart is the
entire point. It is applied afterwards instead, as a separate count, so what it
costs and what it buys are both visible.

    uv run python eval/run_retrieval.py
"""

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuronest.config import settings
from neuronest.embed.cache import CachingEmbedder, EmbeddingCache
from neuronest.embed.local import LocalEmbedder
from neuronest.ingest.chunking import FixedSizeChunker
from neuronest.ingest.loaders import load_directory
from neuronest.retrieve import RetrievedChunk, Retriever
from neuronest.store.chroma import ChromaStore, collection_name_for

EVAL = Path(__file__).parent
CORPUS = EVAL / "corpus"
QUESTIONS = EVAL / "questions.json"
RESULTS = EVAL / "results"

RANKS = (1, 5, 10)

# Below the floor of cosine similarity, so nothing is filtered out and every
# retrieved passage survives to be scored.
NO_THRESHOLD = -1.0


@dataclass(frozen=True)
class Label:
    doc_id: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    kind: str
    topic: str
    labels: tuple[Label, ...]


@dataclass(frozen=True)
class Outcome:
    question: Question
    coverage_rank: int | None
    top_score: float | None
    retrieved: tuple[str, ...]


@dataclass(frozen=True)
class Metrics:
    count: int
    recall: dict[int, float]
    mrr: float


def load_questions(path: Path) -> tuple[dict[str, Any], list[Question]]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    questions = [
        Question(
            id=str(item["id"]),
            text=str(item["question"]),
            kind=str(item["kind"]),
            topic=str(item["topic"]),
            labels=tuple(
                Label(
                    doc_id=str(answer["doc_id"]),
                    char_start=int(answer["char_start"]),
                    char_end=int(answer["char_end"]),
                )
                for answer in item["answers"]
            ),
        )
        for item in data["questions"]
    ]
    return data, questions


def covers(passage: RetrievedChunk, label: Label) -> bool:
    return (
        passage.doc_id == label.doc_id
        and passage.char_start < label.char_end
        and label.char_start < passage.char_end
    )


def coverage_rank(passages: Sequence[RetrievedChunk], labels: Sequence[Label]) -> int | None:
    outstanding = set(range(len(labels)))
    for passage in passages:
        outstanding -= {index for index in outstanding if covers(passage, labels[index])}
        if not outstanding:
            return passage.rank
    return None


def score(outcomes: Sequence[Outcome]) -> Metrics:
    ranks = [outcome.coverage_rank for outcome in outcomes]
    total = len(ranks)
    if not total:
        return Metrics(count=0, recall=dict.fromkeys(RANKS, 0.0), mrr=0.0)

    return Metrics(
        count=total,
        recall={
            rank: sum(1 for found in ranks if found is not None and found <= rank) / total
            for rank in RANKS
        },
        mrr=sum(1.0 / found for found in ranks if found is not None) / total,
    )


def collection_for(model_name: str, fingerprint: str) -> str:
    """Name the collection after the model *and* the chunking that filled it.

    Chunk ids already encode the fingerprint, so two chunking configurations can
    coexist in one collection without colliding — but a query would then search
    across both of them at once and score a sweep against a corpus indexed twice
    over. One collection per configuration keeps each run's index its own.
    """
    return f"{collection_name_for(model_name)}-{fingerprint}"


def build_index(store: ChromaStore, chunker: FixedSizeChunker, corpus: Path) -> tuple[int, int]:
    report = load_directory(corpus)
    if report.errors:
        details = "; ".join(f"{error.source_path.name}: {error.message}" for error in report.errors)
        raise SystemExit(f"corpus did not load cleanly, so nothing was measured: {details}")

    chunks = 0
    for document in report.documents:
        document_chunks = chunker.chunk(document)
        store.ingest(document, document_chunks)
        chunks += len(document_chunks)
    return len(report.documents), chunks


def as_json(metrics: Metrics) -> dict[str, Any]:
    payload: dict[str, Any] = {"questions": metrics.count}
    for rank in RANKS:
        payload[f"recall_at_{rank}"] = round(metrics.recall[rank], 4)
    payload["mrr"] = round(metrics.mrr, 4)
    return payload


def print_table(cohorts: Sequence[tuple[str, Metrics]]) -> None:
    header = f"  {'cohort':<14}{'n':>4}" + "".join(f"{f'R@{rank}':>9}" for rank in RANKS)
    print(f"{header}{'MRR':>9}")
    print(f"  {'-' * (len(header) + 5)}")
    for name, metrics in cohorts:
        scores = "".join(f"{metrics.recall[rank]:>9.3f}" for rank in RANKS)
        print(f"  {name:<14}{metrics.count:>4}{scores}{metrics.mrr:>9.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score retrieval against the labelled set.")
    parser.add_argument("--k", type=int, default=max(RANKS))
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--chroma-path", type=Path, default=settings.chroma_path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--keep-index",
        action="store_true",
        help="Score whatever is already indexed instead of rebuilding it.",
    )
    args = parser.parse_args()

    top_k: int = args.k
    if top_k < max(RANKS):
        raise SystemExit(f"--k must be at least {max(RANKS)} to report Recall@{max(RANKS)}")

    data, questions = load_questions(QUESTIONS)
    # A bare run scores the configuration the set was labelled under; phase 14
    # overrides these to sweep.
    reference = data["reference_chunking"]
    chunk_size: int = args.chunk_size or int(reference["chunk_size"])
    chunk_overlap: int = (
        args.chunk_overlap if args.chunk_overlap is not None else int(reference["chunk_overlap"])
    )

    chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    cache = EmbeddingCache(settings.embedding_cache_path / "embeddings.db")
    embedder = CachingEmbedder(LocalEmbedder(), cache)
    store = ChromaStore(
        embedder,
        path=args.chroma_path,
        collection_name=collection_for(embedder.model_name, chunker.fingerprint),
    )
    # Rebuilt by default rather than reused. Upserting over a collection that
    # already holds these ids perturbs HNSW's approximate neighbours — measured
    # here, one query in a hundred came back with a different rank 1 and a
    # different tail. Two fresh builds are byte-identical, so a number anyone is
    # asked to reproduce has to come from one.
    if not args.keep_index:
        store.reset()

    started = time.perf_counter()
    documents, chunks = build_index(store, chunker, CORPUS)
    index_seconds = time.perf_counter() - started

    retriever = Retriever(store, top_k=top_k, score_threshold=NO_THRESHOLD)
    outcomes: list[Outcome] = []
    query_seconds = 0.0
    for question in questions:
        at = time.perf_counter()
        passages = retriever.retrieve(question.text, k=top_k)
        query_seconds += time.perf_counter() - at
        outcomes.append(
            Outcome(
                question=question,
                coverage_rank=(
                    coverage_rank(passages, question.labels) if question.labels else None
                ),
                top_score=passages[0].score if passages else None,
                retrieved=tuple(passage.chunk_id for passage in passages),
            )
        )

    labelled = [outcome for outcome in outcomes if outcome.question.labels]
    unanswerable = [outcome for outcome in outcomes if outcome.question.kind == "unanswerable"]
    cohorts = [
        ("answerable", score([o for o in outcomes if o.question.kind == "answerable"])),
        ("multi-chunk", score([o for o in outcomes if o.question.kind == "multi_chunk"])),
        ("all labelled", score(labelled)),
    ]

    threshold = settings.score_threshold
    above = sum(1 for o in labelled if o.top_score is not None and o.top_score >= threshold)
    empty = sum(1 for o in unanswerable if o.top_score is None or o.top_score < threshold)

    print(
        f"\n{embedder.model_name} · {chunker.fingerprint} · k={top_k} · "
        f"{documents} documents, {chunks} chunks\n"
    )
    print_table(cohorts)
    print(
        f"\n  at threshold {threshold}: {above}/{len(labelled)} labelled questions clear it, "
        f"{empty}/{len(unanswerable)} unanswerable retrieve nothing"
    )
    print(
        f"  index built in {index_seconds:.1f}s "
        f"(embedding cache {embedder.stats.hit_rate:.0%} hit), "
        f"{len(questions)} queries in {query_seconds:.1f}s "
        f"({query_seconds / len(questions) * 1000:.0f} ms each)\n"
    )

    out: Path = args.out or RESULTS / (
        f"retrieval-{embedder.model_name.split('/')[-1].lower()}-{chunker.fingerprint}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "metrics": {name: as_json(metrics) for name, metrics in cohorts},
                "threshold": {
                    "value": threshold,
                    "labelled_above": above,
                    "labelled_total": len(labelled),
                    "unanswerable_empty": empty,
                    "unanswerable_total": len(unanswerable),
                },
                "questions": [
                    {
                        "id": outcome.question.id,
                        "kind": outcome.question.kind,
                        "topic": outcome.question.topic,
                        "coverage_rank": outcome.coverage_rank,
                        "top_score": (
                            None if outcome.top_score is None else round(outcome.top_score, 4)
                        ),
                        "retrieved": list(outcome.retrieved),
                    }
                    for outcome in outcomes
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  written to {out.relative_to(Path.cwd()) if out.is_absolute() else out}\n")

    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
