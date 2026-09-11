# NeuroNest

A retrieval-augmented generation service: you give it documents, it indexes them, and it answers
questions **only** from the passages it retrieved — refusing outright when the corpus does not contain
the answer.

**Live:** [nestneuroai.vercel.app](https://nestneuroai.vercel.app)

> Early build. The sections below describe the target; commit history shows how far it has got.
> The site is the landing page — the service behind it is still being built, and every figure it
> shows is marked as not yet measured until the evaluation in phases 12-15 produces one.

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

## Licence

MIT — see [LICENSE](LICENSE).
