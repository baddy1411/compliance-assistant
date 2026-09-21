"""Application settings for the compliance RAG data layer.

All values have sensible defaults and can be overridden with environment
variables so containers, CI jobs, and local runs share the same code path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime configuration for ingestion, embeddings, and the vector store."""

    chroma_dir: str = "chroma_db"
    collection_name: str = "compliance_docs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    default_top_k: int = 3
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str | None = None
    llm_model: str = "deepseek-chat"
    kafka_bootstrap: str = "localhost:9092"
    kafka_topic: str = "documents"


def get_settings() -> Settings:
    """Build a :class:`Settings` from the environment, falling back to defaults.

    Supported overrides: CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE,
    CHUNK_OVERLAP, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, KAFKA_BOOTSTRAP.
    """
    return Settings(
        chroma_dir=os.environ.get("CHROMA_DIR", "chroma_db"),
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=_env_int("CHUNK_SIZE", 500),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 50),
        llm_base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        llm_api_key=os.environ.get("LLM_API_KEY") or None,
        llm_model=os.environ.get("LLM_MODEL", "deepseek-chat"),
        kafka_bootstrap=os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092"),
    )
