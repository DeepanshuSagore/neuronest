"""Every chunking configuration, scored against the same labelled set.

This is the experiment the labelled set was built to make possible. Chunking is
the parameter people tune by intuition — "1000 feels right, 200 of overlap for
safety" — and intuition is exactly what a hundred labelled answers can replace.

Two arms, asking different questions.

*Fixed size, three sizes by two overlaps.* Does a bigger window help, and does
overlapping the windows help? Both are cheap to believe and neither is obvious:
a bigger chunk carries more context but dilutes the embedding, and overlap only
does anything when the splitter has to cut mid-paragraph.

*Semantic, at two maximum sizes.* Does cutting where the subject changes beat
cutting at a length? Run at two caps on purpose. Semantic chunks come out larger
than fixed ones, so a single semantic arm confounds "cut in a better place" with
"cut less often" — the 1000-character cap is there to match the reference
configuration's size and separate the two.

Every configuration writes its own stamped result file; this writes one more
naming all of them, which is what the README table is read from.

    uv run python eval/sweep_chunking.py
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_retrieval import (
    QUESTIONS,
    RANKS,
    RESULTS,
    Measurement,
    as_json,
    git_state,
    load_questions,
    measure,
)

from neuronest import __version__
from neuronest.config import settings
from neuronest.embed.cache import CachingEmbedder, EmbeddingCache
from neuronest.embed.local import LocalEmbedder
from neuronest.ingest.chunking import Chunker, FixedSizeChunker, SemanticChunker

SIZES = (500, 1000, 1500)
OVERLAPS = (0, 200)
SEMANTIC_CAPS = (1000, 2000)


def summary_row(measurement: Measurement) -> dict[str, Any]:
    cohorts = dict(measurement.cohorts)
    return {
        "fingerprint": measurement.fingerprint,
        "chunks": measurement.chunks,
        "answerable": as_json(cohorts["answerable"]),
        "multi_chunk": as_json(cohorts["multi-chunk"]),
        "all_labelled": as_json(cohorts["all labelled"]),
        "index_build_seconds": round(measurement.index_seconds, 1),
        "query_mean_ms": round(measurement.query_mean_ms, 1),
        "unanswerable_empty": measurement.unanswerable_empty,
        "result": str(measurement.out.relative_to(RESULTS.parent.parent)),
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    header = f"  {'configuration':<26}{'chunks':>8}" + "".join(f"{f'R@{r}':>8}" for r in RANKS)
    print(f"\n{header}{'MRR':>8}{'build':>8}{'query':>8}")
    print(f"  {'-' * (len(header) + 22)}")
    for row in rows:
        scores = "".join(f"{row['answerable'][f'recall_at_{r}']:>8.3f}" for r in RANKS)
        print(
            f"  {row['fingerprint']:<26}{row['chunks']:>8}{scores}"
            f"{row['answerable']['mrr']:>8.3f}"
            f"{row['index_build_seconds']:>7.1f}s{row['query_mean_ms']:>7.1f}m"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep chunking configurations.")
    parser.add_argument("--k", type=int, default=max(RANKS))
    parser.add_argument("--chroma-path", type=Path, default=settings.chroma_path)
    parser.add_argument("--out", type=Path, default=RESULTS / "sweep-chunking.json")
    args = parser.parse_args()

    data, questions = load_questions(QUESTIONS)
    cache = EmbeddingCache(settings.embedding_cache_path / "embeddings.db")
    embedder = CachingEmbedder(LocalEmbedder(), cache)

    plan: list[tuple[Chunker, dict[str, float]]] = [
        (
            FixedSizeChunker(chunk_size=size, chunk_overlap=overlap),
            {"chunk_size": size, "chunk_overlap": overlap},
        )
        for size in SIZES
        for overlap in OVERLAPS
    ]
    plan += [
        (
            SemanticChunker(embedder, max_chunk_chars=cap),
            {
                "breakpoint_percentile": 95,
                "buffer_size": 1,
                "min_chunk_chars": 200,
                "max_chunk_chars": cap,
            },
        )
        for cap in SEMANTIC_CAPS
    ]

    git = git_state()
    rows: list[dict[str, Any]] = []
    try:
        for chunker, chunk_params in plan:
            rows.append(
                summary_row(
                    measure(
                        embedder=embedder,
                        chunker=chunker,
                        chunk_params=chunk_params,
                        data=data,
                        questions=questions,
                        top_k=args.k,
                        chroma_path=args.chroma_path,
                        out=None,
                        keep_index=False,
                        git=git,
                    )
                )
            )
    finally:
        cache.close()

    print_summary(rows)
    best = max(rows, key=lambda row: row["answerable"]["recall_at_5"])
    print(f"\n  best Recall@5 over the answerable set: {best['fingerprint']}\n")

    commit, dirty = git
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "neuronest": __version__,
                "git_commit": commit,
                "git_dirty": dirty,
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "embedding_model": embedder.model_name,
                "top_k": args.k,
                "reference": data["reference_chunking"]["fingerprint"],
                "configurations": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    out: Path = args.out
    shown = out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out
    print(f"  written to {shown}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
