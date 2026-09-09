# NeuroNest

A retrieval-augmented generation service: you give it documents, it indexes them, and it answers
questions **only** from the passages it retrieved — refusing outright when the corpus does not contain
the answer.

**Live:** [nestneuroai.vercel.app](https://nestneuroai.vercel.app)

> Early build. The sections below describe the target; commit history shows how far it has got.
> The site is the landing page — the service behind it is still being built, and every figure it
> shows is marked as not yet measured until the evaluation in phases 12-15 produces one.

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

## Run the service

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
 "passages": [{"chunk_id": "5d87034c...-0000-210c6428", "source_path": "corpus/scaling-laws.md",
               "char_start": 0, "char_end": 310, "score": 0.53, "rank": 1,
               "text": "Loss scales as a power law with model size, dataset size and compute..."}],
 "refused": false, "score_threshold": 0.25}
```

Ask something the corpus does not cover and you get an empty result, not the least-bad passage:

```json
{"question": "What was the training run's electricity bill?",
 "passages": [], "refused": true, "score_threshold": 0.25}
```

That is a `200`, not an error. Refusing is the behaviour this service exists to demonstrate — phase 10
never calls the language model on a refusal, so it cannot invent an answer out of unrelated text.

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
