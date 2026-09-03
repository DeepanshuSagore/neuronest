"""Env-driven application settings.

Every value that phases 14 and 15 will sweep — chunk size, overlap, embedding
model, ``top_k`` — lives here from the first commit that needs it. A tunable
hard-coded at its call site turns a config sweep into a refactor, and the
experiments are the point of this project.

Two dotenv files are read, in order: ``.env`` then ``.env.local``. The second
wins, so shared defaults can be committed as ``.env.example`` while personal
secrets stay in an ignored local override.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage ---
    # Where the persistent Chroma collection lives. Derived data: deleting it
    # costs a re-ingest, not a restore.
    chroma_path: Path = Path(".chroma")

    # --- embeddings ---
    # A sentence-transformers model id, resolved locally. Phase 15 swaps this
    # for two others and compares recall against index size and query latency.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- generation ---
    # Blank is a valid, supported state: everything up to retrieval works
    # without a key, and the service starts and serves /health regardless.
    # Generation is the only thing that degrades.
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # --- chunking ---
    # Characters, not tokens — RecursiveCharacterTextSplitter counts characters
    # and pretending otherwise would make the phase 14 table lie.
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    # --- retrieval ---
    top_k: int = Field(default=5, gt=0)

    # --- logging ---
    log_level: LogLevel = "INFO"

    @model_validator(mode="after")
    def _overlap_fits_inside_chunk(self) -> "Settings":
        """Reject overlap >= size at startup rather than at ingest time.

        LangChain raises on this combination partway through splitting, by
        which point the traceback points at the splitter instead of at the
        parameter sweep that produced the pair.
        """
        if self.chunk_overlap >= self.chunk_size:
            msg = (
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
            raise ValueError(msg)
        return self


settings = Settings()
