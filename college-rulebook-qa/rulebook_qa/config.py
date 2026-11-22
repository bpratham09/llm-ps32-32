"""Central configuration. Every value can be overridden with an environment variable or the .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


# Provider choices
#   LLM_PROVIDER   : "openai" (any OpenAI-compatible API) or "gemini"
#   EMBED_PROVIDER : "fastembed" (free, runs locally, no key) or "gemini"
_LLM_PROVIDER = _env("LLM_PROVIDER", "openai").lower()
_EMBED_PROVIDER = _env("EMBED_PROVIDER", "fastembed").lower()
_DEFAULT_CHAT = {"openai": "gpt-6.1-sol", "gemini": "gemini-3.5-flash-lite"}
_DEFAULT_EMBED = {"fastembed": "BAAI/bge-small-en-v1.5", "gemini": "models/gemini-embedding-2"}
KEY_ENV = {"openai": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY"}


@dataclass(frozen=True)
class Settings:
    # Models
    llm_provider: str = _LLM_PROVIDER
    embed_provider: str = _EMBED_PROVIDER
    chat_model: str = _env("CHAT_MODEL", _DEFAULT_CHAT.get(_LLM_PROVIDER, "gpt-6.1-sol"))
    embed_model: str = _env("EMBED_MODEL", _DEFAULT_EMBED.get(_EMBED_PROVIDER, "BAAI/bge-small-en-v1.5"))
    openai_base_url: str = _env("OPENAI_BASE_URL", "")  # empty = official OpenAI endpoint
    temperature: float = float(_env("TEMPERATURE", "0"))

    # Chunking and retrieval
    chunk_size: int = int(_env("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "150"))
    top_k: int = int(_env("TOP_K", "4"))
    history_turns: int = int(_env("HISTORY_TURNS", "3"))

    # Paths
    index_dir: Path = ROOT_DIR / _env("INDEX_DIR", "faiss_index")
    sample_pdf: Path = ROOT_DIR / "data" / "sample_college_handbook.pdf"
    eval_questions: Path = ROOT_DIR / "data" / "eval_questions.json"
    eval_output_dir: Path = ROOT_DIR / "evaluation"
    feedback_file: Path = ROOT_DIR / "logs" / "feedback.csv"


settings = Settings()


def key_env_name() -> str:
    """Name of the environment variable holding the LLM provider's API key."""
    return KEY_ENV.get(settings.llm_provider, "OPENAI_API_KEY")


def has_api_key() -> bool:
    needed = {key_env_name()}
    if settings.embed_provider == "gemini":
        needed.add("GOOGLE_API_KEY")
    return all(os.getenv(k) for k in needed)
