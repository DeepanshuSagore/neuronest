"""Settings behaviour: defaults, dotenv precedence, and validation.

These tests ``chdir`` into a temporary directory rather than passing
``_env_file`` overrides, because dotenv paths resolve against the working
directory. Constructing ``Settings`` with an override would exercise a
different code path from the one that runs in production and would keep
passing if the real file lookup broke.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuronest.config import Settings

SETTING_ENV_VARS = (
    "CHROMA_PATH",
    "EMBEDDING_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in SETTING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def test_defaults_apply_when_no_dotenv_is_present(tmp_path: Path) -> None:
    assert not list(tmp_path.iterdir())

    settings = Settings()

    assert settings.chroma_path == Path(".chroma")
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.groq_api_key == ""
    assert settings.groq_model == "openai/gpt-oss-120b"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.top_k == 5
    assert settings.log_level == "INFO"


def test_a_blank_groq_key_is_a_valid_state(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("GROQ_API_KEY=\n")

    assert Settings().groq_api_key == ""


def test_values_load_from_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "CHROMA_PATH=/tmp/index\nCHUNK_SIZE=512\nTOP_K=8\nLOG_LEVEL=DEBUG\n"
    )

    settings = Settings()

    assert settings.chroma_path == Path("/tmp/index")
    assert settings.chunk_size == 512
    assert settings.top_k == 8
    assert settings.log_level == "DEBUG"


def test_env_local_overrides_env(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("GROQ_API_KEY=from-shared\nTOP_K=3\n")
    (tmp_path / ".env.local").write_text("GROQ_API_KEY=from-local\n")

    settings = Settings()

    assert settings.groq_api_key == "from-local"
    assert settings.top_k == 3


def test_the_process_environment_wins_over_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("TOP_K=3\n")
    monkeypatch.setenv("TOP_K", "11")

    assert Settings().top_k == 11


@pytest.mark.parametrize(
    ("field", "value"),
    [("CHUNK_SIZE", "0"), ("TOP_K", "0"), ("LOG_LEVEL", "LOUD")],
)
def test_out_of_range_values_are_rejected(
    field: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(field, value)

    with pytest.raises(ValidationError):
        Settings()


def test_overlap_must_be_smaller_than_chunk_size(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("CHUNK_SIZE=200\nCHUNK_OVERLAP=200\n")

    with pytest.raises(ValidationError, match="must be smaller than"):
        Settings()
