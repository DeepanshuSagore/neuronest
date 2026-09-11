"""Generation, driven through the real SDK against a transport that never leaves the process.

The client is stubbed at the transport rather than at the client object, so
every test still exercises the SDK's own request building and response parsing.
A test that replaces ``chat.completions.create`` with a lambda proves only that
the lambda was called, and would keep passing after the provider changed the
shape of what it returns.
"""

import json
from collections.abc import Callable

import httpx
import pytest
from groq import Groq

from neuronest.generate import REFUSAL_TOKEN, Generator
from neuronest.retrieve import RetrievedChunk

Handler = Callable[[httpx.Request], httpx.Response]


def passage(text: str, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc-{rank:04d}-abcd1234",
        doc_id="doc",
        text=text,
        char_start=0,
        char_end=len(text),
        source_path="/srv/corpus/handbook.md",
        score=0.5,
        rank=rank,
    )


def completion(content: str, finish_reason: str = "stop") -> dict[str, object]:
    """One chat completion, shaped the way the API actually returns it."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def generator_over(handler: Handler) -> Generator:
    # max_retries=0 so a call is one request and counting them stays honest.
    client = Groq(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return Generator(model_name="test-model", client=client)


def answering(content: str, finish_reason: str = "stop") -> Handler:
    return lambda _request: httpx.Response(200, json=completion(content, finish_reason))


@pytest.fixture
def recorded() -> list[httpx.Request]:
    return []


def recording(seen: list[httpx.Request], content: str = "An answer [1].") -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion(content))

    return handler


def sent_messages(request: httpx.Request) -> list[dict[str, str]]:
    body = json.loads(request.content)
    return [dict(message) for message in body["messages"]]


# --- answering ---------------------------------------------------------------


def test_it_answers_from_the_passages() -> None:
    generator = generator_over(answering("The relay runs for up to 94 minutes [1]."))

    result = generator.generate("How long can it run?", [passage("94 minutes on batteries")])

    assert result.answer == "The relay runs for up to 94 minutes [1]."
    assert result.refused is False
    assert result.note is None


def test_the_passages_are_supplied_as_numbered_evidence(recorded: list[httpx.Request]) -> None:
    """Numbered by rank, which is what makes a [2] in the answer resolvable."""
    generator = generator_over(recording(recorded))

    generator.generate(
        "How long can it run?",
        [passage("94 minutes on batteries", 1), passage("42 kilobits per second", 2)],
    )

    prompt = sent_messages(recorded[0])[-1]["content"]
    assert "[1] 94 minutes on batteries" in prompt
    assert "[2] 42 kilobits per second" in prompt
    assert "How long can it run?" in prompt


def test_the_model_is_told_the_passages_are_its_only_source(recorded: list[httpx.Request]) -> None:
    generator = generator_over(recording(recorded))

    generator.generate("How long?", [passage("94 minutes")])

    system = sent_messages(recorded[0])[0]
    assert system["role"] == "system"
    assert "only" in system["content"].lower()
    assert REFUSAL_TOKEN in system["content"]


def test_the_prompt_carries_no_server_filesystem_paths(recorded: list[httpx.Request]) -> None:
    """The model does not need them, and a cited filename is not a citation."""
    generator = generator_over(recording(recorded))

    generator.generate("How long?", [passage("94 minutes")])

    assert "/srv/corpus" not in recorded[0].content.decode("utf-8")


def test_it_asks_for_the_configured_model(recorded: list[httpx.Request]) -> None:
    generator = generator_over(recording(recorded))

    generator.generate("How long?", [passage("94 minutes")])

    assert json.loads(recorded[0].content)["model"] == "test-model"


# --- declining to answer -----------------------------------------------------


def test_a_model_refusal_is_reported_as_a_refusal() -> None:
    """Passages retrieved, and none of them answer the question.

    The adjacent-but-absent case phase 11 builds twenty questions around:
    retrieval clears the threshold on subject matter alone, and only the model
    can see that the answer is not actually there.
    """
    generator = generator_over(answering(REFUSAL_TOKEN))

    result = generator.generate("What is the payroll schedule?", [passage("thermal limits")])

    assert result.refused is True
    assert result.answer is None
    assert result.note is not None


def test_a_refusal_wrapped_in_prose_still_counts() -> None:
    generator = generator_over(answering(f"I'm sorry, {REFUSAL_TOKEN}."))

    assert generator.generate("q", [passage("text")]).refused is True


def test_an_empty_completion_is_not_an_answer() -> None:
    """A reasoning model can spend its whole budget thinking and return nothing."""
    generator = generator_over(answering("", finish_reason="length"))

    result = generator.generate("How long?", [passage("94 minutes")])

    assert result.answer is None
    assert result.refused is False
    assert "length" in (result.note or "")


def test_an_empty_retrieval_never_reaches_the_model(recorded: list[httpx.Request]) -> None:
    """The refusal the whole service is built around, and it costs nothing.

    Asserted on the transport: with no passages there is no HTTP request at
    all, so the model cannot have invented an answer out of unrelated text
    and no tokens were spent discovering that.
    """
    generator = generator_over(recording(recorded))

    result = generator.generate("What is the relay operator payroll schedule?", [])

    assert recorded == []
    assert result.refused is True
    assert result.answer is None
    assert result.note is not None


def test_refusing_needs_no_key_either(recorded: list[httpx.Request]) -> None:
    """Retrieving nothing is a complete answer, so a missing key cannot mask it."""
    result = Generator(api_key="").generate("What is the payroll schedule?", [])

    assert result.refused is True
    assert "corpus" in (result.note or "")


def test_without_a_key_it_says_so_instead_of_answering() -> None:
    """A blank key is a supported state: everything but this step still works."""
    generator = Generator(api_key="", model_name="test-model")

    result = generator.generate("How long?", [passage("94 minutes")])

    assert generator.available is False
    assert result.answer is None
    assert result.refused is False
    assert "GROQ_API_KEY" in (result.note or "")


# --- when the provider fails -------------------------------------------------


def unreachable(_request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused")


def test_a_provider_outage_returns_a_note_rather_than_raising() -> None:
    """The passages are the expensive half of the work and they are still correct."""
    generator = generator_over(unreachable)

    result = generator.generate("How long can it run?", [passage("94 minutes")])

    assert result.answer is None
    assert "Could not generate" in (result.note or "")


def test_a_provider_failure_is_not_recorded_as_a_refusal() -> None:
    """Phase 13 scores refusal accuracy, and an outage counted as a refusal inflates it.

    Declining and failing are different events. The service declines when the
    evidence does not support an answer; it fails when it could not ask. Folding
    the second into the first would make an unreliable provider look like good
    judgement.
    """
    generator = generator_over(unreachable)

    assert generator.generate("How long?", [passage("94 minutes")]).refused is False


def erroring(status: int) -> Handler:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "upstream is having a day"}})

    return handler


def test_a_server_error_degrades_the_same_way() -> None:
    generator = generator_over(erroring(500))

    result = generator.generate("How long?", [passage("94 minutes")])

    assert result.answer is None
    assert result.note is not None


def test_a_rate_limit_degrades_the_same_way() -> None:
    generator = generator_over(erroring(429))

    assert generator.generate("How long?", [passage("94 minutes")]).answer is None
