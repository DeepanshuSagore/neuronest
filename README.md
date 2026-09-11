# NeuroNest

A retrieval-augmented generation service: you give it documents, it indexes them, and it answers
questions **only** from the passages it retrieved — refusing outright when the corpus does not contain
the answer.

**Live:** [nestneuroai.vercel.app](https://nestneuroai.vercel.app)

> Early build. The sections below describe the target; commit history shows how far it has got.
> Retrieval is measured — every figure in [Measured retrieval](#measured-retrieval) traces to a
> checked-in result file. Faithfulness and refusal accuracy are not. The landing page marks
> unmeasured figures rather than inventing them, and has not yet been filled in from this run.

## Quickstart

```bash
git clone https://github.com/DeepanshuSagore/neuronest
cd neuronest
docker compose up
```

That is all of it. No Python, no `uv`, no account — ingestion, retrieval and refusal all run locally,
and the model weights ship inside the image, so the first request does not wait on a download. The
API comes up on [localhost:8000](http://localhost:8000) with interactive docs at
[/docs](http://localhost:8000/docs).

```bash
curl -X POST localhost:8000/ingest -F "file=@paper.pdf"
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question":"What does the paper claim about scaling laws?"}'
```

Writing an answer out of those passages is the one step that calls a hosted model. Put a free
[Groq](https://console.groq.com) key in `.env.local` — which is gitignored, and which compose passes
through — and `/query` comes back with a written, cited answer:

```bash
echo 'GROQ_API_KEY=your-key-here' >> .env.local
```

Without one, everything else still works: the service ingests, retrieves and refuses, and `/query`
returns the passages with a note saying why there is no written answer.

The index lives in a named volume, so it survives `docker compose down`. To throw it away, either
empty the corpus through the API with `curl -X DELETE localhost:8000/corpus`, or take the volume
with it using `docker compose down -v`.

### What the image costs

Measured on `linux/arm64` under Colima (4 CPU, 6 GB) with `all-MiniLM-L6-v2`:

| Metric | Value |
|---|---|
| Image size, unpacked | **1.90 GB** — sum of layer sizes from `docker history` |
| — Python environment | 1.57 GB of that, torch alone 652 MB |
| — Embedding weights | 88 MB, baked into a layer |
| — Precompiled bytecode | 337 MB across 16,477 `.pyc` files |
| Cold start, weights baked | **10.1–13.4 s** to a healthy `/health`, over three runs |
| Cold start, weights fetched | 39.6–43.0 s over two runs — what not baking them costs |
| Warm `/health` | 37–55 ms |

`docker image ls` reports 2.46 GB for the same image on this setup; it and `docker history` count
differently, and the layer sum is the one quoted above.

Two decisions account for most of the size.

**torch comes from the PyTorch CPU index, not PyPI.** On Linux the default wheel declares the entire
CUDA runtime — `cuda-toolkit`, `cudnn`, `nccl`, `triton` — for a GPU no container here will ever
have. Redirecting it drops thirteen `nvidia-*` packages plus `triton` from the resolution, and it is
worth more than every other size optimisation in the image put together. macOS keeps the ordinary
wheel behind a platform marker, so local development is unaffected.

**Bytecode is compiled during the build.** It costs 337 MB and buys the cold-start number: the venv
is read-only to the runtime user, so modules compiled on first import would be re-parsed on every
cold start and never cached. That is the right trade for phase 20, where the host sleeps and wakes
cold.

A container started with `--network none` still answers `/health` with `"status": "ok"` — which is
how the claim that nothing is fetched at runtime gets checked rather than asserted.

## Why this one is different

There are thousands of RAG chatbots on GitHub and their READMEs all say the same thing: "it works."
This repo is built around the part almost everyone skips — **measuring retrieval quality separately
from generation quality**, against a hand-labelled evaluation set written *before* any tuning
happened.

Most RAG failures are retrieval failures that get blamed on the model. Telling those two apart
requires numbers, and the numbers require a labelled set you did not write to flatter your current
configuration.

## Measured retrieval

```bash
uv run python eval/run_retrieval.py
```

`all-MiniLM-L6-v2`, fixed 1000/200 chunking, 40 documents, 5,810 chunks, k=10, no score threshold
applied. Every figure below is read from
[`eval/results/retrieval-all-minilm-l6-v2-fixed-1000-200.json`](eval/results/retrieval-all-minilm-l6-v2-fixed-1000-200.json),
which carries the configuration and the corpus and question digests that produced it.

| Cohort | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| Answerable | 70 | 0.429 | 0.729 | 0.800 | 0.551 |
| Answer needs two chunks | 10 | 0.000 | 0.200 | 0.400 | 0.112 |
| All labelled | 80 | 0.375 | 0.662 | 0.750 | 0.496 |

A question is scored correct when a retrieved passage **overlaps the labelled character span** — not
when a chunk id matches, because chunk ids encode the chunker's settings and the chunking experiment
changes them. For a two-chunk question the rank counted is the one at which *both* spans have been
seen, which is why its Recall@1 is 0.000: that is arithmetic, not failure. One passage cannot cover
two spans, so the earliest such a question can succeed is rank 2.

Retrieval alone costs about 1 ms per query against a warm index. Building the index takes 42 s cold
and 4.7 s once the embedding cache is populated, which is what makes re-running the sweep free.

### The score threshold does nothing here, and that is the finding

Out of 20 unanswerable questions, **0 retrieve nothing** at the configured threshold of 0.25. All 80
labelled questions clear it too. The threshold is inert on this corpus:

| | Weakest | Strongest |
|---|---|---|
| Top-1 score, labelled questions | **0.513** | 0.859 |
| Top-1 score, unanswerable questions | 0.347 | **0.716** |

The strongest unanswerable question scores well above the weakest answerable one, so **no threshold
separates them**. 0.25 was not guessed — it was measured, but against a three-document sample where
answerable questions scored 0.311–0.711 and absent subjects 0.004–0.180. On forty real documents that
gap closes completely, because a question about QUIC still matches forty documents' worth of prose
about HTTP over TCP.

The practical consequence is that refusal currently rests entirely on the model declining to answer
from passages that do not contain the answer, not on the score floor — which is the behaviour phase
13 measures.

### These numbers understate recall by a measured amount

All 20 misses were read by hand, all ten retrieved passages each. In **6 of them retrieval returned
the fact, from a location the label does not name**:

- **q033** — "what size limit applies to DNS messages over UDP?" The label points at RFC 1035's
  parameter table. Retrieval returned RFC 6891's prose, RFC 8484's quotation of the same rule, and
  RFC 1035's own sentence stating it. Three correct answers, none of them the labelled one.
- **q048** — "maximum length of a multipart boundary?" The label says *"no longer than 70
  characters"*; retrieval returned *"consists of 1 to 70 characters"* from the same document.

Counting those six as hits gives Recall@10 of 0.871 answerable and 0.825 overall. **The published
numbers are the unadjusted ones.** The labels are not being edited to raise a score: the protocol in
[`eval/README.md`](eval/README.md) says a question may be corrected for being *wrong*, not for being
*hard*, and re-labelling only the questions that happened to miss would bias the set in exactly one
direction. The bias is stated instead, and it is constant across the comparisons in phases 14 and 15,
which score every configuration against these same labels.

The other 14 are genuine retrieval failures — q001 asks what a 408 means and gets back the passage
defining status-code *classes*; q022 asks how Basic auth builds its credential and gets OAuth client
passwords from a different RFC entirely.

Numbers reproduce: the index is rebuilt on every run, and four consecutive rebuilds produced
identical metrics. Only the ordering of the retrieved tail for two *unanswerable* queries varies,
because HNSW search is approximate; neither carries a label, so nothing above moves.

## Stack

**Service** — Python 3.12 · FastAPI · pydantic v2 · LangChain · ChromaDB · sentence-transformers ·
Groq · Docker · pytest · ruff + mypy (strict)

**Web** — Next.js · Tailwind · anime.js, in [`frontend/`](frontend)

## Web

A landing page for the service lives in `frontend/`. It renders one content model, so the copy in
the hero, the demo transcript and the results table cannot drift apart, and every published figure
is gated behind a placeholder flag until the evaluation in phases 12–15 actually produces it.

```bash
cd frontend
bun install
bun dev          # http://localhost:3000
```

## What it is not

- **Not multi-tenant.** One corpus, one index, no per-user isolation.
- **Not authenticated.** There are no accounts and no API keys of its own.
- **English only.** No multilingual embedding or evaluation.
- **Not a general chatbot.** It has no memory between questions and no knowledge outside the corpus.
  Asking it something the documents do not cover gets you a refusal, by design, not a best guess.

## Run from source

Without Docker, for development. Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run uvicorn neuronest.api.app:app --reload
```

Interactive docs at [localhost:8000/docs](http://localhost:8000/docs).

### Ingest and query, end to end

```bash
# one document, uploaded
curl -X POST localhost:8000/ingest -F "file=@paper.pdf"
# {"documents_ingested":1,"chunks_written":14,"failures":[]}

# or a whole folder already on the server
curl -X POST localhost:8000/ingest/local \
  -H 'Content-Type: application/json' \
  -d '{"path":"./corpus"}'
```

Re-ingesting the same document updates it rather than duplicating it, and files that cannot be
indexed come back described rather than silently skipped:

```json
{"documents_ingested": 39, "chunks_written": 512,
 "failures": [{"source": "corpus/scan.pdf", "code": "no_text_layer",
               "message": "No extractable text across 12 page(s) — it is almost certainly a scan..."}]}
```

Ask a question:

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does the paper claim about scaling laws?"}'
```

```json
{"question": "What does the paper claim about scaling laws?",
 "answer": "Loss falls as a power law in model size, dataset size and compute [1].",
 "passages": [{"chunk_id": "5d87034c...-0000-210c6428", "source_path": "corpus/scaling-laws.md",
               "char_start": 0, "char_end": 310, "score": 0.53, "rank": 1,
               "text": "Loss scales as a power law with model size, dataset size and compute..."}],
 "refused": false, "note": null, "score_threshold": 0.25}
```

The bracketed numbers are passage ranks: `[1]` is the passage listed at rank 1, so every claim leads
back to a chunk id you can go and read. The passages in the prompt are the model's only permitted
source — it is not asked to answer the question, it is asked to answer it *from these*.

There are two different ways to get no answer, and the response tells them apart.

**Nothing was relevant enough.** No passage cleared the score threshold, so there was no evidence to
reason over and the language model was never called at all:

```json
{"question": "Who won the 1998 World Cup final?",
 "answer": null, "passages": [], "refused": true,
 "note": "Nothing in the corpus was relevant enough to this question to answer from...",
 "score_threshold": 0.25}
```

**Something was relevant, and it still did not contain the answer.** This is the harder case, and the
one most services get wrong: the index finds passages on the right subject, because subject matter is
all a vector search can see. Only the model can read them and notice the answer is not there.

```json
{"question": "What was the training run's electricity bill?",
 "answer": null, "passages": [{"score": 0.48, "rank": 1, "...": "..."}], "refused": true,
 "note": "The retrieved passages do not contain an answer to this question...",
 "score_threshold": 0.25}
```

Both are a `200`, not an error. Refusing is the behaviour this service exists to demonstrate, and the
passages come back either way because they are the evidence for the refusal.

Add `"generate": false` to retrieve passages without answering. That is what the evaluation sweeps
run: they measure retrieval over the whole question set many times, and generating an answer on every
one would put a bill on a measurement that has to stay free to repeat.

### Operations

```bash
curl localhost:8000/health    # checks Chroma and the loaded model, not a hardcoded ok
curl localhost:8000/stats     # corpus size and the configuration that produced it
curl -X DELETE localhost:8000/corpus
```

`/health` reports what it actually read, and goes `unhealthy` with a reason when the store or the
model is not there:

```json
{"status": "ok", "store_reachable": true,
 "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dimensions": 384,
 "collection": "nn-sentence-transformers-all-minilm-l6-v2-e9e2c881", "chunks_indexed": 512}
```

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed automatically from
`.python-version`.

```bash
uv sync          # create .venv and install dependencies
uv run ruff check .
uv run mypy
uv run pytest
```

The evaluation set and its harness live in [`eval/`](eval), and neither needs an API key — the
embedder runs locally and no answer is generated while scoring retrieval.

```bash
uv run python eval/validate_questions.py   # every label still points at the text it claims
uv run python eval/run_retrieval.py        # Recall@k and MRR, table plus a stamped result file
uv run python eval/build_corpus.py         # refetch the 40 documents from rfc-editor.org
```

## Licence

MIT — see [LICENSE](LICENSE).
