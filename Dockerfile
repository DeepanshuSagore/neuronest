# syntax=docker/dockerfile:1

# Two stages for one reason: the toolchain that installs ~900MB of wheels does
# not need to ship with them. The builder resolves the environment, the runtime
# copies it and takes nothing else — no uv, no build cache, no wheel archives.

FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    # Precompiled at build time rather than written on first import: the venv is
    # read-only to the runtime user, so uncompiled modules would re-parse on
    # every cold start and never cache. Costs image size, buys startup — the
    # right trade for phase 20, where the host sleeps and wakes cold.
    UV_COMPILE_BYTECODE=1

WORKDIR /build

# Dependencies install from the lockfile alone, in a layer only the lockfile can
# invalidate. Editing source then rebuilds in seconds instead of pulling torch
# again. __init__.py is mounted because the version is dynamic and hatchling
# reads it from there to produce the project's metadata.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=src/neuronest/__init__.py,target=src/neuronest/__init__.py \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# --no-editable installs the package itself into the venv, so /opt/venv is
# self-contained and the runtime stage needs no copy of the source tree.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Fetch the weights now, so no request ever waits on a download.
#
# Loaded through SentenceTransformer rather than fetched with a snapshot
# download, for two reasons. The repository also carries ONNX, OpenVINO and
# TensorFlow copies of the same weights, and a blind snapshot takes all of
# them; loading through the library takes only the files this service reads.
# And it fails the build if the model cannot be loaded, rather than at the
# first query.
#
# The model id comes from the settings module so the baked weights cannot drift
# from the configured default. No .env reaches the build context, so this
# resolves to the documented default.
ENV HF_HOME=/opt/models
RUN /opt/venv/bin/python -c "from neuronest.config import settings; \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer(settings.embedding_model)"


FROM python:3.12-slim-bookworm AS runtime

# A fixed uid, not a floating one: a bind-mounted index directory has to have
# predictable ownership on whatever host mounts it.
#
# The home directory is not decoration. Chroma writes an anonymous telemetry id
# under ~/.cache on the first connection, so a user without a writable home
# fails at store.count() — reported by /health as an unreachable store, which
# is a confusing way to discover a missing directory.
RUN groupadd --system --gid 1001 neuronest \
 && useradd --system --uid 1001 --gid neuronest --create-home neuronest

COPY --from=builder /opt/venv /opt/venv

# Owned by the runtime user rather than root. The weights are read-only in
# practice, but huggingface_hub takes a lock file inside the cache when it
# resolves a model, and it cannot do that in a directory it may not write.
# Keeping the cache writable also means overriding EMBEDDING_MODEL still works
# — it downloads, the way an unbaked model has to.
COPY --from=builder --chown=neuronest:neuronest /opt/models /opt/models

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/models \
    CHROMA_PATH=/data/chroma \
    EMBEDDING_CACHE_PATH=/data/embedding-cache

# The index and the embedding cache are the only paths the service writes to,
# and they are the only paths the runtime user owns. Both are derived from the
# corpus, which is why one volume covers both: losing it costs a re-ingest.
RUN install -d -o neuronest -g neuronest /data

WORKDIR /srv
USER neuronest
EXPOSE 8000

# Hits the real /health, which opens the collection and asks the embedder for
# the width it actually produces. A check that only proved the port was open
# would go green while the store was unreachable.
HEALTHCHECK --start-period=180s --interval=30s --timeout=20s --retries=3 \
    CMD ["python", "-c", "import json,sys,urllib.request; sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=15))['status'] == 'ok' else 1)"]

CMD ["uvicorn", "neuronest.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
