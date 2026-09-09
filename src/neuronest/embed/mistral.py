"""The hosted embedder, behind the same protocol as the local one.

Exists so phase 15 has something real to measure the local default against.
Three local models make a dull comparison; local against hosted produces an
actual tradeoff — recall against query latency, index size and the ability to
reproduce a number offline — which is the judgement that section is for.

Not the default, and deliberately so. Requiring it would put a network call in
front of every ingest and every query, mean the test suite needs a secret, and
tie a published Recall@5 to a model the vendor can retire underneath it.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from neuronest.config import settings
from neuronest.embed.base import Vector

if TYPE_CHECKING:
    from mistralai.client import Mistral

# The API accepts up to 512 inputs per call. Batching to the limit is what keeps
# a corpus-sized run inside the free tier's per-minute request budget: 1,500
# chunks is three requests this way and 1,500 requests one at a time.
MAX_INPUTS_PER_REQUEST = 512


class MistralEmbedder:
    """Embeds via Mistral's hosted API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        key = api_key if api_key is not None else settings.mistral_api_key
        if not key:
            msg = (
                "MISTRAL_API_KEY is not set. The hosted embedder is opt-in; "
                "the local sentence-transformers embedder is the default and needs no key."
            )
            raise ValueError(msg)

        self._api_key = key
        self._model_name = (
            model_name if model_name is not None else settings.mistral_embedding_model
        )
        requested = batch_size if batch_size is not None else settings.embedding_batch_size
        self._batch_size = min(requested, MAX_INPUTS_PER_REQUEST)
        self._client: Mistral | None = None
        self._dimensions: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Measured, not assumed.

        The dimension depends on the model and, for the dim128 and dim256
        variants, on which one was configured. One short embed costs a handful
        of tokens and removes a hard-coded 1024 that would silently corrupt an
        index the day a different variant is selected.
        """
        if self._dimensions is None:
            self._dimensions = len(self.embed(["dimension probe"])[0])
        return self._dimensions

    def _connected(self) -> "Mistral":
        if self._client is None:
            from mistralai.client import Mistral

            self._client = Mistral(api_key=self._api_key)
        return self._client

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []

        client = self._connected()
        vectors: list[Vector] = []

        for offset in range(0, len(texts), self._batch_size):
            window = list(texts[offset : offset + self._batch_size])
            response = client.embeddings.create(model=self._model_name, inputs=window)

            data = response.data if response is not None else None
            if data is None or len(data) != len(window):
                msg = (
                    f"Mistral returned {0 if data is None else len(data)} embeddings "
                    f"for {len(window)} inputs."
                )
                raise RuntimeError(msg)

            for item in data:
                if item.embedding is None:
                    msg = "Mistral returned an entry with no embedding."
                    raise RuntimeError(msg)
                vectors.append([float(value) for value in item.embedding])

        return vectors
