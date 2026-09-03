# NeuroNest

A retrieval-augmented generation service: you give it documents, it indexes them, and it answers
questions **only** from the passages it retrieved — refusing outright when the corpus does not contain
the answer.

> Early build. The sections below describe the target; commit history shows how far it has got.

## Why this one is different

There are thousands of RAG chatbots on GitHub and their READMEs all say the same thing: "it works."
This repo is built around the part almost everyone skips — **measuring retrieval quality separately
from generation quality**, against a hand-labelled evaluation set written *before* any tuning
happened.

Most RAG failures are retrieval failures that get blamed on the model. Telling those two apart
requires numbers, and the numbers require a labelled set you did not write to flatter your current
configuration.

## Stack

Python 3.12 · FastAPI · pydantic v2 · LangChain · ChromaDB · sentence-transformers · Groq · Docker ·
pytest · ruff + mypy (strict)

## What it is not

- **Not multi-tenant.** One corpus, one index, no per-user isolation.
- **Not authenticated.** There are no accounts and no API keys of its own.
- **English only.** No multilingual embedding or evaluation.
- **Not a general chatbot.** It has no memory between questions and no knowledge outside the corpus.
  Asking it something the documents do not cover gets you a refusal, by design, not a best guess.

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
