"""Tests for ingest.chunk_document (real contract: character budgets).

Chunk ids look like ``"<doc_id>:::<idx:04d>"``; per-chunk origin metadata
(``title``, ``source_path``, ``chunk_index``) lives in ``chunk.metadata``.
"""

from __future__ import annotations

from itertools import pairwise

from conftest import make_document

from ingest import chunk_document

LOREM = ("The quick brown fox jumps over the lazy dog. " * 60).strip()

# Every word unique, so shared words between consecutive chunks can only come
# from the overlap window.
UNIQUE_WORDS = " ".join(f"word{i}" for i in range(600))


def test_chunk_sizes_bounded() -> None:
    """No chunk exceeds the character budget."""
    doc = make_document(text=LOREM)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    for chunk in chunks:
        assert 0 < len(chunk.text) <= 200


def test_overlap_continuity() -> None:
    """Consecutive chunks share the overlap window's text."""
    doc = make_document(text=UNIQUE_WORDS)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    for previous, nxt in pairwise(chunks):
        tail = previous.text.split()[-8:]
        head = nxt.text.split()[:30]
        assert set(tail) & set(head), (
            "consecutive chunks share no overlapping words"
        )


def test_empty_text_returns_no_chunks() -> None:
    """Empty or whitespace-only documents produce no chunks."""
    assert chunk_document(make_document(text="")) == []
    assert chunk_document(make_document(text="   \n  ")) == []


def test_metadata_preserved() -> None:
    """Origin metadata (title/source_path/chunk_index) lands on every chunk."""
    doc = make_document(
        doc_id="policy",
        title="Policy Title",
        source_path="policy.md",
        text=LOREM,
    )
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    assert chunks
    for index, chunk in enumerate(chunks):
        assert chunk.doc_id == "policy"
        assert chunk.metadata["title"] == "Policy Title"
        assert chunk.metadata["source_path"] == "policy.md"
        assert chunk.metadata["chunk_index"] == index


def test_chunk_id_format() -> None:
    """Chunk ids are unique and follow the '<doc_id>:::<idx:04d>' format."""
    doc = make_document(doc_id="gdpr_principles", text=LOREM)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(set(ids)) == len(ids)  # unique
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"gdpr_principles:::{index:04d}"
