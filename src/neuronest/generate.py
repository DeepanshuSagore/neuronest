"""Turning retrieved passages into an answer, or declining to write one.

The prompt hands the model the retrieved passages as its only permitted
evidence, and that constraint is the entire mechanism. A model asked a question
with no constraint answers from its training data, and an answer that did not
come from the corpus is indistinguishable, to a reader, from one that did.

Two different things make this decline, and the response tells them apart.
Retrieval can return nothing, in which case there is no evidence to reason over
and the model is never called. Or retrieval can return passages that turn out
not to answer the question — which is exactly what an adjacent-but-absent
question looks like: the index finds something on the right subject and it does
not contain the answer. Nothing before the model can see that, so the model is
told to say so, and phase 13 scores how often it does.

A provider failure is not an error either. The passages are the expensive half
of the work and they are still correct, so a failed call returns them with a
note instead of a 500 and the caller keeps the evidence.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from neuronest.config import settings
from neuronest.retrieve import RetrievedChunk

if TYPE_CHECKING:
    from groq import Groq

# What the model says instead of guessing. Distinctive and uppercase so that
# testing for it cannot be tripped by ordinary prose, and matched anywhere in
# the reply rather than as the whole of it, because a model that has been asked
# for one token still occasionally wraps it in a sentence.
REFUSAL_TOKEN = "NOT_IN_CONTEXT"

# Generous because the configured default is a reasoning model: a probe that
# returned an eleven-character answer spent 37 of its 49 completion tokens
# thinking first. A budget sized for the answer alone gets consumed by the
# reasoning and returns empty content with finish_reason "length", which reads
# as a provider failure and is really a configuration mistake.
MAX_COMPLETION_TOKENS = 1024

# Deterministic, so that re-running the phase 13 faithfulness judge over the
# same corpus scores the same answers rather than newly sampled ones.
TEMPERATURE = 0.0

# The SDK's own default is minutes long, which is a sensible default for a batch
# script and far too long for something sitting behind an HTTP request.
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 2

SYSTEM_PROMPT = (
    "You answer questions using only the numbered passages given in the user "
    "message. They are your only permitted source of fact.\n"
    "\n"
    "Rules:\n"
    "- Use only what the passages state. Do not add anything from your own "
    "knowledge, even when you are confident it is correct.\n"
    "- Cite the passage each fact came from by its bracketed number, like [1]. "
    "A sentence stating a fact carries at least one citation.\n"
    f"- If the passages do not contain the answer, reply with {REFUSAL_TOKEN} "
    "and nothing else. Related or adjacent information is not an answer.\n"
    "- Answer in at most four sentences."
)

NO_KEY_NOTE = (
    "Generation is unavailable: GROQ_API_KEY is not configured. "
    "The retrieved passages are below."
)
DECLINED_NOTE = (
    "The retrieved passages do not contain an answer to this question, "
    "so no answer was generated."
)


class GeneratedAnswer(BaseModel):
    """What came back from the attempt to answer.

    ``answer`` and ``note`` are mutually exclusive and one of them is always
    set: either there is an answer, or there is a reason there is not.
    """

    model_config = ConfigDict(frozen=True)

    answer: str | None
    #: True when the service declined to answer rather than failed to. A caller
    #: distinguishes the two kinds of refusal by whether passages were retrieved
    #: at all: none means retrieval found nothing, some means the model read
    #: them and said they do not answer the question.
    refused: bool
    #: Why there is no answer, in a sentence, when there is no answer.
    note: str | None = None


class Generator:
    """Answers from passages via Groq, or explains why it did not."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        client: "Groq | None" = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.groq_api_key
        self._model_name = model_name if model_name is not None else settings.groq_model
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def available(self) -> bool:
        """Whether answering is possible at all.

        A blank key is a supported state, not a misconfiguration: ingestion,
        retrieval and refusal all work without one, and the service starts and
        serves regardless. Only this step degrades.
        """
        return self._client is not None or bool(self._api_key)

    def _connected(self) -> "Groq":
        if self._client is None:
            from groq import Groq

            self._client = Groq(
                api_key=self._api_key,
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )
        return self._client

    def generate(self, question: str, passages: Sequence[RetrievedChunk]) -> GeneratedAnswer:
        """Answer ``question`` from ``passages`` and nothing else."""
        if not self.available:
            return GeneratedAnswer(answer=None, refused=False, note=NO_KEY_NOTE)

        from groq import GroqError

        try:
            content, finish_reason = self._complete(question, passages)
        except GroqError as exc:
            return GeneratedAnswer(
                answer=None,
                refused=False,
                note=(
                    f"Could not generate an answer: {type(exc).__name__}. "
                    f"The retrieved passages are below."
                ),
            )

        if REFUSAL_TOKEN in content:
            return GeneratedAnswer(answer=None, refused=True, note=DECLINED_NOTE)

        if not content.strip():
            return GeneratedAnswer(
                answer=None,
                refused=False,
                note=(
                    f"The model returned no answer text (finish_reason: {finish_reason}). "
                    f"The retrieved passages are below."
                ),
            )

        return GeneratedAnswer(answer=content.strip(), refused=False)

    def _complete(self, question: str, passages: Sequence[RetrievedChunk]) -> tuple[str, str]:
        response = self._connected().chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_evidence_prompt(question, passages)},
            ],
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )

        if not response.choices:
            return "", "no_choices"

        choice = response.choices[0]
        return choice.message.content or "", choice.finish_reason


def build_evidence_prompt(question: str, passages: Sequence[RetrievedChunk]) -> str:
    """Lay the passages out as numbered evidence, then ask the question.

    Numbered by rank, which is what makes a citation resolvable: ``[2]`` in an
    answer is the passage the response reports at rank 2, so a reader follows it
    back to a chunk id without the model having to be trusted to repeat one.

    Source paths are deliberately left out. They are server filesystem paths,
    the model does not need them to answer, and putting them in the prompt
    invites it to cite a filename instead of a passage number.
    """
    evidence = "\n\n".join(f"[{passage.rank}] {passage.text}" for passage in passages)
    return f"PASSAGES:\n{evidence}\n\nQUESTION: {question}"
