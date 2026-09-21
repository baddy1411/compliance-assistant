"""Shared fixtures for the compliance-assistant test suite.

Everything here is offline: embeddings are deterministic fakes derived from
SHA-256, vector stores live in tmp dirs, and the LLM is stubbed. The project
root (parent of ``tests/``) is added to ``sys.path`` so the sibling layers
(``ingest``, ``vectorstore``, ``rag``, ``api`` ...) import exactly as they do
in production.

Assumed data-layer contracts (see NOTES_OPS.md -- the data-layer sibling
owns ``ingest.py`` / ``vectorstore.py`` / ``embeddings.py`` / ``config.py``):

* ``ingest.Document(doc_id, title, text, source_path, metadata)``
* ``ingest.Chunk(doc_id, chunk_id, text, metadata)`` -- chunk ids look like
  ``"<doc_id>:::<idx:04d>"``; ``metadata`` carries ``title``,
  ``source_path`` and ``chunk_index``
* ``vectorstore.ComplianceVectorStore(persist_dir, collection_name,
  embedding_fn)`` with ``add_chunks`` / ``query`` / ``count`` / ``reset``
* ``embeddings.get_embedding_fn()`` returns
  ``callable(texts: list[str]) -> list[list[float]]``
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest import Chunk, Document


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic 16-dim normalized pseudo-random vectors seeded by text.

    Identical texts produce identical vectors (cosine similarity 1.0), which
    makes "most similar doc" assertions meaningful without any model.
    """
    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [byte / 255.0 for byte in digest[:16]]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        vectors.append([value / norm for value in values])
    return vectors


@pytest.fixture()
def embedding_fn() -> Any:
    """Deterministic fake embedding function (dim 16, no network)."""
    return _fake_embed


@pytest.fixture()
def tmp_chroma_dir(tmp_path: Path) -> str:
    """Fresh Chroma persistence directory for one test."""
    directory = tmp_path / "chroma"
    directory.mkdir()
    return str(directory)


SAMPLE_TEXT = (
    "The GDPR requires lawfulness, fairness and transparency in processing. "
    "Data minimisation limits collection to what is necessary. "
    "The EU AI Act classifies recruitment systems as high-risk. "
    "Northwind retains customer data for twenty-four months. "
)


def make_document(
    doc_id: str = "doc_1",
    text: str = SAMPLE_TEXT,
    title: str = "Sample",
    source_path: str = "sample.md",
    metadata: dict[str, Any] | None = None,
) -> Document:
    """Build a sample ``ingest.Document``."""
    return Document(
        doc_id=doc_id,
        title=title,
        text=text,
        source_path=source_path,
        metadata=dict(metadata or {}),
    )


def make_chunk(
    doc_id: str = "doc_1",
    index: int = 0,
    text: str = "sample chunk text",
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    """Build a sample ``ingest.Chunk`` (chunk_index lives in metadata)."""
    meta = dict(metadata or {})
    meta.setdefault("title", "Sample")
    meta.setdefault("source_path", "sample.md")
    meta["chunk_index"] = index
    return Chunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}:{index}",
        text=text,
        metadata=meta,
    )
