"""Faithfulness and refusal accuracy, with the judge itself put on trial.

Phase 12 scored retrieval on its own. This scores the half that comes after it:
given passages, does the answer stay inside them, and when the passages do not
contain an answer does the service say so instead of writing one anyway.

Three numbers come out, and the third is the one that makes the first two worth
reading.

*Faithfulness* is the share of generated answers whose every factual claim is
supported by the passages that answer was given. An LLM judge decides it,
against the rubric below, which is written down rather than improvised per call.

*Refusal accuracy* is the share of the twenty unanswerable questions the service
declined. It is reported split by whether the subject is even named anywhere in
the corpus, because those two are not the same test: declining a question about
BGP, a word that appears nowhere, needs no judgement at all, while declining one
about QUIC — which RFC 9110 mentions without explaining — does.

*Judge-versus-human agreement* is the share of a twenty-item sample where the
judge's verdict matches one I wrote by hand. Reporting a judge's score without
validating the judge is the most common mistake in this whole category: an
unvalidated judge is an unmeasured instrument, and a faithfulness percentage
produced by one means nothing. The hand labels live in ``human_sample.json`` and
are written from the answers file *before* the judge has run, which is what
makes the comparison worth anything — ``--sample-sheet`` prints exactly what
there is to label, and it contains no verdicts.

    uv run python eval/run_faithfulness.py --answers-only   # generate, no judge
    uv run python eval/run_faithfulness.py --sample-sheet    # the 20 to label
    uv run python eval/run_faithfulness.py                   # judge and score
"""

import argparse
import json
import random
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_retrieval import (
    CORPUS,
    EVAL,
    QUESTIONS,
    RESULTS,
    Question,
    build_index,
    collection_for,
    digest_of,
    git_state,
    load_questions,
)

from neuronest import __version__
from neuronest.config import settings
from neuronest.embed.cache import CachingEmbedder, EmbeddingCache
from neuronest.embed.local import LocalEmbedder
from neuronest.generate import Generator
from neuronest.ingest.chunking import FixedSizeChunker
from neuronest.ingest.loaders import load_directory
from neuronest.retrieve import RetrievedChunk, Retriever
from neuronest.store.chroma import ChromaStore

HUMAN_SAMPLE = EVAL / "human_sample.json"

# Fixed so the twenty items under human review are the same twenty on every
# run. Picking them afresh each time would let a disappointing agreement number
# be improved by re-rolling, which is the same failure as re-labelling a
# question because it turned out hard.
SAMPLE_SEED = 13
SAMPLE_SIZE = 20

# The free tier enforces three budgets and the one that binds is not the one the
# headers advertise. x-ratelimit-limit-tokens reports 8,000 per minute and
# x-ratelimit-limit-requests 1,000 per day; only a 429 body names the ceiling
# that actually stops a run, "tokens per day (TPD): Limit 200000". At roughly
# 2,300 charged per judge call a seventy-question pass spends about 80% of a
# day, so two full runs cannot both finish in one — which is a fact about what
# phases 14 and 15 can sweep, not merely about how long this takes.
TOKENS_PER_MINUTE = 8000
TOKENS_PER_DAY = 200_000

# Reserved per call and charged in full by the limiter whether the judge uses it
# or not: a 13-token prompt asking for 512 is billed "Requested 525". Pacing on
# usage.total_tokens instead runs light by exactly the unused remainder, and the
# debt only surfaces as a refusal near the end of a long run.
MAX_COMPLETION_TOKENS = 512

# A 429 against a budget that refills continuously is a wait, not a dead end,
# and the response states its length. Guessing instead is what spent six fixed
# backoffs on one question and returned no verdict for it.
RATE_LIMIT_RETRIES = 6
RATE_LIMIT_BACKOFF_SECONDS = 20
RATE_LIMIT_MAX_WAIT_SECONDS = 900

RUBRIC = """You check whether an answer is supported by the passages it was given.

You are judging support and nothing else. Do not judge whether the answer is \
true in the world, whether it is well written, or whether it is complete.

Mark the answer "unsupported" if any of the following is true:
- it states a fact the passages do not state and that does not follow directly \
from them, even when that fact is correct in the world;
- it cites a passage number that does not contain what it is cited for;
- it contradicts a passage.

Mark the answer "supported" when every factual claim it makes is stated by the \
passages or follows directly from them. Connective language, hedging and \
restatement of the question are not claims.

Reply with exactly one line of JSON and nothing else:
{"verdict": "supported" | "unsupported", "reason": "<one short sentence>"}"""

VERDICTS = ("supported", "unsupported")


@dataclass(frozen=True)
class Passage:
    """A retrieved passage as the answers file records it.

    Deliberately narrower than ``RetrievedChunk``: these four fields are what
    judging and reporting use, and they are exactly what the file stores, so it
    round-trips without loss. Widening this to the full chunk would put document
    offsets in a record that never reads them, and server filesystem paths in a
    file that gets committed.
    """

    rank: int
    chunk_id: str
    score: float
    text: str


@dataclass(frozen=True)
class Answered:
    """One question taken all the way through the service's own path."""

    question: Question
    answer: str | None
    refused: bool
    note: str | None
    passages: tuple[Passage, ...]

    @property
    def retrieval_was_empty(self) -> bool:
        return not self.passages


@dataclass(frozen=True)
class Verdict:
    supported: bool | None
    reason: str


def evidence_block(passages: Sequence[Passage]) -> str:
    return "\n\n".join(f"[{passage.rank}] {passage.text}" for passage in passages)


def judge_prompt(item: Answered) -> str:
    return (
        f"PASSAGES:\n{evidence_block(item.passages)}\n\n"
        f"QUESTION: {item.question.text}\n\n"
        f"ANSWER: {item.answer}"
    )


def read_verdict(raw: str) -> Verdict:
    """Turn the judge's reply into a verdict, or report that it did not give one.

    An unparseable reply is its own outcome rather than a default. Folding it
    into "unsupported" would understate faithfulness and folding it into
    "supported" would overstate it; both would hide a judge that had stopped
    answering the question it was asked.
    """
    text = raw.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            verdict = str(parsed.get("verdict", "")).strip().lower()
            if verdict in VERDICTS:
                return Verdict(
                    supported=verdict == "supported",
                    reason=str(parsed.get("reason", "")).strip(),
                )

    lowered = text.lower()
    if "unsupported" in lowered:
        return Verdict(supported=False, reason="recovered from unparsed reply")
    if "supported" in lowered:
        return Verdict(supported=True, reason="recovered from unparsed reply")
    return Verdict(supported=None, reason=f"unparseable judge reply: {text[:120]!r}")


def stated_delay(exc: Any, attempt: int) -> float:
    """How long the provider says to wait, falling back to a widening guess.

    Every one of these budgets refills continuously rather than at a fixed hour,
    so a 429 carries a real deadline — "try again in 25.488s" — and the response
    header repeats it. A fixed backoff either gives up while the budget is still
    filling or sleeps through headroom it already has; the provider is the only
    party that knows which.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    stated = headers.get("retry-after")
    if stated is not None:
        try:
            return min(float(stated) + 1.0, RATE_LIMIT_MAX_WAIT_SECONDS)
        except (TypeError, ValueError):
            pass
    return float(RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1))


class Judge:
    """Scores one answer against the passages it was given."""

    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._client: Any = None

    def _connected(self) -> Any:
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self._api_key, timeout=60.0, max_retries=3)
        return self._client

    def verdict(self, item: Answered) -> Verdict:
        """Score one answer, waiting out the rate limiter rather than giving up.

        Seventy of these run back to back against a free tier with a per-minute
        token budget, so exhausting it is the expected case rather than an
        exceptional one. The SDK's own retries are tuned for a request that has
        a user waiting on it and give up far too early here; a batch script can
        afford to sit out a minute. Getting this wrong is not a slow run, it is
        a wrong number: a rate-limited judge returns no verdicts, the answers it
        never saw drop out of the denominator, and what survives is a
        faithfulness score computed over a handful of lucky calls.
        """
        from groq import GroqError, RateLimitError

        for attempt in range(RATE_LIMIT_RETRIES):
            started = time.perf_counter()
            try:
                response = self._connected().chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": RUBRIC},
                        {"role": "user", "content": judge_prompt(item)},
                    ],
                    temperature=0.0,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                )
            except RateLimitError as exc:
                if attempt == RATE_LIMIT_RETRIES - 1:
                    return Verdict(supported=None, reason="judge call failed: RateLimitError")
                waited = stated_delay(exc, attempt)
                print(f"      rate limited, waiting {waited:.0f}s", file=sys.stderr, flush=True)
                time.sleep(waited)
                continue
            except GroqError as exc:
                return Verdict(supported=None, reason=f"judge call failed: {type(exc).__name__}")

            usage = response.usage
            charged = (usage.prompt_tokens if usage else 0) + MAX_COMPLETION_TOKENS
            self._pace(charged, time.perf_counter() - started)

            if not response.choices:
                return Verdict(supported=None, reason="judge returned no choices")
            return read_verdict(response.choices[0].message.content or "")

        return Verdict(supported=None, reason="judge call failed: RateLimitError")

    def _pace(self, tokens: int, elapsed: float) -> None:
        """Hold this call's share of the minute before returning.

        A call costing n tokens is entitled to n/budget of a minute; if it came
        back faster than that, the difference is slept off. Self-tuning, because
        the entitlement is computed from what the call actually cost rather than
        from a guess about what it might.
        """
        share = 60.0 * tokens / TOKENS_PER_MINUTE
        if share > elapsed:
            time.sleep(share - elapsed)


def as_passages(chunks: Sequence[RetrievedChunk]) -> tuple[Passage, ...]:
    return tuple(
        Passage(rank=c.rank, chunk_id=c.chunk_id, score=round(c.score, 4), text=c.text)
        for c in chunks
    )


def answer_all(
    questions: Sequence[Question], retriever: Retriever, generator: Generator
) -> list[Answered]:
    answered: list[Answered] = []
    for question in questions:
        passages = retriever.retrieve(question.text)
        result = generator.generate(question.text, passages)
        answered.append(
            Answered(
                question=question,
                answer=result.answer,
                refused=result.refused,
                note=result.note,
                passages=as_passages(passages),
            )
        )
    return answered


def load_answers(path: Path, questions: Sequence[Question]) -> list[Answered]:
    """Read back a previous answering run.

    What makes the two-stage protocol usable: a blind sample sheet is printed
    from this file, labelled by hand, and only then judged. Regenerating between
    those steps would cost a quarter of an hour and a hundred more provider
    calls to reproduce answers that are already on disk.
    """
    by_id = {question.id: question for question in questions}
    stored = json.loads(path.read_text(encoding="utf-8"))
    return [
        Answered(
            question=by_id[str(item["id"])],
            answer=item["answer"],
            refused=bool(item["refused"]),
            note=item["note"],
            passages=tuple(
                Passage(
                    rank=int(p["rank"]),
                    chunk_id=str(p["chunk_id"]),
                    score=float(p["score"]),
                    text=str(p["text"]),
                )
                for p in item["passages"]
            ),
        )
        for item in stored["answers"]
    ]


def topics_named_in_corpus(questions: Sequence[Question]) -> dict[str, bool]:
    """Which unanswerable questions are about a subject the corpus even mentions.

    The split this produces is the whole reason refusal accuracy is not reported
    as one number. A question about a word that appears nowhere can be declined
    on vocabulary alone; one about a subject the corpus names in passing cannot.
    """
    flattened = " ".join(
        " ".join(document.text.split()).lower()
        for document in load_directory(CORPUS).documents
    )
    return {
        question.id: question.topic.lower() in flattened
        for question in questions
        if question.kind == "unanswerable"
    }


def prior_verdicts(path: Path, judge_model: str, answers_digest: str) -> dict[str, Verdict]:
    """Usable verdicts from an earlier run of this exact configuration.

    The free tier allows a thousand requests a day, refilling at one every 86
    seconds, and a run that dies two thirds of the way through has already spent
    its share. Re-judging an answer whose verdict is on disk buys nothing and is
    what exhausted the budget the first time: three restarts of a seventy-call
    pass cost more than two hundred requests and produced one incomplete file.

    Reuse is gated on everything that could change a verdict — the judge, the
    rubric, the questions and the answers themselves. A verdict carried across a
    change to any of those is a number credited to a run that never produced it,
    which is worse than paying for it again.
    """
    if not path.exists():
        return {}
    stored = json.loads(path.read_text(encoding="utf-8"))
    config = stored.get("config", {})
    reusable = (
        config.get("judge_model") == judge_model
        and config.get("rubric") == RUBRIC
        and config.get("questions", {}).get("sha256") == digest_of([QUESTIONS])
        and config.get("answers", {}).get("sha256") == answers_digest
    )
    if not reusable:
        return {}
    return {
        entry["id"]: Verdict(supported=entry["supported"], reason=entry["reason"])
        for entry in stored.get("verdicts", [])
        if entry.get("supported") is not None
    }


def sample_ids(answered: Sequence[Answered]) -> list[str]:
    """The items a human labels, chosen the same way on every run."""
    judgeable = sorted(item.question.id for item in answered if item.answer is not None)
    size = min(SAMPLE_SIZE, len(judgeable))
    return sorted(random.Random(SAMPLE_SEED).sample(judgeable, size))


def agreement(human: dict[str, str], machine: dict[str, bool | None]) -> dict[str, Any] | None:
    """Raw agreement and Cohen's kappa between the two sets of labels.

    Kappa as well as the raw percentage because raw agreement flatters any pair
    of labellers who mostly say the same thing: if ninety per cent of answers
    are faithful, two labellers who both say "supported" reflexively agree ninety
    per cent of the time while demonstrating nothing. Kappa subtracts the
    agreement that chance alone would produce at those base rates.
    """
    pairs = [
        (human[qid] == "supported", machine[qid])
        for qid in sorted(human)
        if qid in machine and machine[qid] is not None
    ]
    if not pairs:
        return None

    total = len(pairs)
    observed = sum(1 for mine, theirs in pairs if mine == theirs) / total

    human_yes = sum(1 for mine, _ in pairs if mine) / total
    judge_yes = sum(1 for _, theirs in pairs if theirs) / total
    expected = human_yes * judge_yes + (1 - human_yes) * (1 - judge_yes)
    kappa = 1.0 if expected == 1.0 else (observed - expected) / (1 - expected)

    return {
        "items": total,
        "raw_agreement": round(observed, 4),
        "cohens_kappa": round(kappa, 4),
        "human_supported": round(human_yes, 4),
        "judge_supported": round(judge_yes, 4),
        "disagreements": [
            qid
            for qid in sorted(human)
            if qid in machine
            and machine[qid] is not None
            and (human[qid] == "supported") != machine[qid]
        ],
    }


def print_sample_sheet(answered: Sequence[Answered], chosen: Sequence[str]) -> None:
    by_id = {item.question.id: item for item in answered}
    print(
        f"\n{len(chosen)} items to label by hand, seed {SAMPLE_SEED}. "
        f"No judge verdicts appear below.\n"
    )
    for qid in chosen:
        item = by_id[qid]
        print("=" * 96)
        print(f"{qid}  [{item.question.kind}/{item.question.topic}]  {item.question.text}")
        print(f"\nANSWER: {item.answer}\n")
        for passage in item.passages:
            # Printed in full, deliberately. The judge reads the whole passage,
            # so a human labelling a truncated one is answering an easier
            # question and the agreement between them measures the truncation as
            # much as the judge.
            print(f"  [{passage.rank}] ({passage.score:.3f}) {' '.join(passage.text.split())}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Score faithfulness and refusal accuracy.")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--chroma-path", type=Path, default=settings.chroma_path)
    parser.add_argument("--judge-model", type=str, default=settings.judge_model)
    parser.add_argument("--answers", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--answers-only", action="store_true")
    parser.add_argument("--sample-sheet", action="store_true")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Generate answers again instead of reusing the ones already on disk.",
    )
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="Score every answer again instead of carrying verdicts from the result file.",
    )
    args = parser.parse_args()

    data, questions = load_questions(QUESTIONS)
    reference = data["reference_chunking"]
    chunk_size: int = args.chunk_size or int(reference["chunk_size"])
    chunk_overlap: int = (
        args.chunk_overlap if args.chunk_overlap is not None else int(reference["chunk_overlap"])
    )

    chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    generator = Generator()
    answers_path: Path = args.answers or RESULTS / f"answers-{chunker.fingerprint}.json"

    # Built here rather than inside the branch below, because the result file is
    # stamped with the embedding model whichever branch ran. Both are cheap:
    # the cache is a SQLite handle and the embedder loads its weights lazily, on
    # the first call that actually needs them.
    cache = EmbeddingCache(settings.embedding_cache_path / "embeddings.db")
    embedder = CachingEmbedder(LocalEmbedder(), cache)

    documents, chunks = 0, 0
    answer_seconds = 0.0
    top_k, threshold = settings.top_k, settings.score_threshold

    if answers_path.exists() and not args.refresh:
        answered = load_answers(answers_path, questions)
        stored = json.loads(answers_path.read_text(encoding="utf-8"))
        top_k, threshold = int(stored["top_k"]), float(stored["score_threshold"])
        # Counted rather than left at zero: the stamp on the result file has to
        # describe the corpus these answers came from whichever branch produced
        # them. Loading and chunking is seconds; only embedding is not.
        loaded = load_directory(CORPUS).documents
        documents = len(loaded)
        chunks = sum(len(chunker.chunk(document)) for document in loaded)
        print(f"\n  {len(answered)} answers reused from {answers_path.name} (--refresh to redo)")
    else:
        if not generator.available:
            raise SystemExit("GROQ_API_KEY is not set, so no answers can be generated.")

        store = ChromaStore(
            embedder,
            path=args.chroma_path,
            collection_name=collection_for(embedder.model_name, chunker.fingerprint),
        )
        store.reset()
        documents, chunks = build_index(store, chunker, CORPUS)

        # The real service path, threshold and all. Phase 12 measured retrieval
        # with the floor switched off on purpose; here it stays on, because
        # refusing is part of what is being scored.
        retriever = Retriever(store)
        top_k, threshold = retriever.top_k, retriever.score_threshold

        started = time.perf_counter()
        answered = answer_all(questions, retriever, generator)
        answer_seconds = time.perf_counter() - started

        answers_path.parent.mkdir(parents=True, exist_ok=True)
        answers_path.write_text(
            json.dumps(
                {
                    "generation_model": generator.model_name,
                    "top_k": top_k,
                    "score_threshold": threshold,
                    "answers": [
                        {
                            "id": item.question.id,
                            "kind": item.question.kind,
                            "topic": item.question.topic,
                            "question": item.question.text,
                            "answer": item.answer,
                            "refused": item.refused,
                            "note": item.note,
                            "passages": [
                                {
                                    "rank": passage.rank,
                                    "chunk_id": passage.chunk_id,
                                    "score": passage.score,
                                    "text": passage.text,
                                }
                                for passage in item.passages
                            ],
                        }
                        for item in answered
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"\n  {len(answered)} answers written to {answers_path.name} "
            f"in {answer_seconds:.0f}s"
        )

    chosen = sample_ids(answered)
    if args.sample_sheet:
        print_sample_sheet(answered, chosen)
        cache.close()
        return 0
    if args.answers_only:
        cache.close()
        return 0

    out: Path = args.out or RESULTS / f"faithfulness-{chunker.fingerprint}.json"
    answers_digest = digest_of([answers_path])
    carried = {} if args.rejudge else prior_verdicts(out, args.judge_model, answers_digest)
    if carried:
        print(f"  {len(carried)} verdicts carried from {out.name} (--rejudge to redo)")

    judge = Judge(args.judge_model, settings.groq_api_key)
    to_score = sum(
        1 for item in answered if item.answer is not None and item.question.id not in carried
    )
    print(f"  judging {to_score} answers at roughly three a minute\n")

    verdicts: dict[str, Verdict] = {}
    reused = 0
    started = time.perf_counter()
    for item in answered:
        if item.answer is None:
            continue
        known = carried.get(item.question.id)
        if known is not None:
            verdicts[item.question.id] = known
            reused += 1
            continue
        verdicts[item.question.id] = judge.verdict(item)
        # On stderr so it stays out of the report on stdout. Pacing makes this
        # a twenty-minute run that otherwise prints nothing until it ends, and
        # a silent run is indistinguishable from a hung one.
        print(
            f"    {len(verdicts) - reused}/{to_score} {item.question.id} "
            f"{verdicts[item.question.id].supported}",
            file=sys.stderr,
            flush=True,
        )
    judge_seconds = time.perf_counter() - started

    judged = [v for v in verdicts.values() if v.supported is not None]
    supported = [v for v in judged if v.supported]
    unjudged = len(verdicts) - len(judged)

    labelled = [item for item in answered if item.question.kind != "unanswerable"]
    unanswerable = [item for item in answered if item.question.kind == "unanswerable"]
    named = topics_named_in_corpus(questions)

    declined = [item for item in unanswerable if item.refused]
    over_refused = [item for item in labelled if item.refused]
    hard = [item for item in unanswerable if named[item.question.id]]
    easy = [item for item in unanswerable if not named[item.question.id]]

    human: dict[str, str] = {}
    if HUMAN_SAMPLE.exists():
        human = dict(json.loads(HUMAN_SAMPLE.read_text(encoding="utf-8"))["labels"])
    scored = agreement(human, {qid: v.supported for qid, v in verdicts.items()})

    faithfulness = len(supported) / len(judged) if judged else 0.0
    print(f"\n  generation {generator.model_name} · judge {args.judge_model}")
    print(f"  corpus {documents} documents, {chunks} chunks · k={top_k}")
    print(f"\n  faithfulness        {faithfulness:.3f}  ({len(supported)}/{len(judged)} answers)")
    if unjudged:
        print(f"  unjudged            {unjudged} (the judge gave no usable verdict)")
    print(
        f"  refusal accuracy    {len(declined) / len(unanswerable):.3f}  "
        f"({len(declined)}/{len(unanswerable)} unanswerable questions declined)"
    )
    print(
        f"    subject named       {sum(1 for i in hard if i.refused)}/{len(hard)}  "
        f"(the corpus mentions it without answering)"
    )
    print(
        f"    subject absent      {sum(1 for i in easy if i.refused)}/{len(easy)}  "
        f"(the word appears nowhere)"
    )
    print(
        f"  over-refusal        {len(over_refused) / len(labelled):.3f}  "
        f"({len(over_refused)}/{len(labelled)} answerable questions declined)"
    )
    if scored:
        print(
            f"\n  judge vs human      raw {scored['raw_agreement']:.3f} · "
            f"kappa {scored['cohens_kappa']:.3f}  over {scored['items']} hand-labelled items"
        )
    else:
        print(f"\n  judge vs human      not computed: {HUMAN_SAMPLE.name} holds no usable labels")

    commit, dirty = git_state()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": {
                    "neuronest": __version__,
                    "git_commit": commit,
                    "git_dirty": dirty,
                    "generation_model": generator.model_name,
                    "judge_model": args.judge_model,
                    "embedding": {
                        "model": embedder.model_name,
                        "dimensions": embedder.dimensions,
                    },
                    "chunking": {
                        "strategy": chunker.fingerprint.split("-", 1)[0],
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "fingerprint": chunker.fingerprint,
                    },
                    "retrieval": {
                        "top_k": top_k,
                        "score_threshold": threshold,
                        "threshold_applied_to_metrics": True,
                    },
                    "corpus": {
                        "documents": documents,
                        "chunks": chunks,
                        "sha256": digest_of(sorted(CORPUS.glob("*.txt"))),
                    },
                    "questions": {"sha256": digest_of([QUESTIONS])},
                    # Pins the verdicts to the answers they were passed. Without
                    # it a resumed run could carry a verdict onto a different
                    # answer to the same question and still look consistent.
                    "answers": {"sha256": answers_digest},
                    "rubric": RUBRIC,
                },
                "run": {
                    "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                    "answer_seconds": round(answer_seconds, 1),
                    "judge_seconds": round(judge_seconds, 1),
                    "verdicts_carried": reused,
                    "verdicts_scored": len(verdicts) - reused,
                },
                "metrics": {
                    "faithfulness": round(faithfulness, 4),
                    "answers_judged": len(judged),
                    "answers_supported": len(supported),
                    "answers_unjudged": unjudged,
                    "refusal_accuracy": round(len(declined) / len(unanswerable), 4),
                    "refusal_subject_named": f"{sum(1 for i in hard if i.refused)}/{len(hard)}",
                    "refusal_subject_absent": f"{sum(1 for i in easy if i.refused)}/{len(easy)}",
                    "over_refusal": round(len(over_refused) / len(labelled), 4),
                    "over_refused_ids": [i.question.id for i in over_refused],
                },
                "judge_vs_human": scored,
                "verdicts": [
                    {
                        "id": qid,
                        "supported": verdicts[qid].supported,
                        "reason": verdicts[qid].reason,
                        "in_human_sample": qid in chosen,
                    }
                    for qid in sorted(verdicts)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    shown = out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out
    print(f"\n  written to {shown}\n")
    cache.close()

    # An incomplete run is a failure, not a result. Answers the judge never
    # scored drop out of the denominator rather than counting against the score,
    # so a rate-limited run reports a *higher* faithfulness over a handful of
    # lucky calls — the one direction of error nobody checks. Exiting non-zero
    # is what stops that number reaching a README.
    if unjudged:
        print(
            f"  INCOMPLETE: {unjudged} of {len(verdicts)} answers carry no verdict, so the "
            f"faithfulness above was computed over a subset and is not publishable.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
