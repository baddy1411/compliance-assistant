"""Tests for ComplianceVectorStore using deterministic fake embeddings.

Fully offline: the store persists to a tmp dir and embeds with the fake
SHA-256 embedding function from conftest. Assumes the constructor accepts
``persist_dir``, ``collection_name`` and ``embedding_fn`` keywords, and that
``query`` returns RetrievedChunk objects with ``doc_id`` / ``score``.
See NOTES_OPS.md.
"""

from __future__ import annotations

from conftest import make_chunk

from vectorstore import ComplianceVectorStore

DOCS = [
    (
        "gdpr",
        (
            "The GDPR sets out principles of lawfulness, fairness and "
            "transparency for personal data."
        ),
    ),
    (
        "ai_act",
        (
            "The EU AI Act classifies certain AI systems as high-risk, "
            "including those used in hiring."
        ),
    ),
    (
        "northwind",
        (
            "Northwind retains customer personal data for 24 months after the "
            "relationship ends."
        ),
    ),
]


def make_store(tmp_chroma_dir: str, embedding_fn, name: str = "test"):
    """Build a store on a tmp dir with fake embeddings."""
    return ComplianceVectorStore(
        persist_dir=tmp_chroma_dir, collection_name=name, embedding_fn=embedding_fn
    )


def _seeded(store: ComplianceVectorStore):
    chunks = [
        make_chunk(doc_id=doc_id, index=0, text=text) for doc_id, text in DOCS
    ]
    assert store.add_chunks(chunks) == 3
    return chunks


def test_add_chunks_returns_count(tmp_chroma_dir, embedding_fn) -> None:
    """add_chunks returns the number of chunks added."""
    store = make_store(tmp_chroma_dir, embedding_fn)
    assert store.add_chunks([make_chunk(index=i) for i in range(3)]) == 3
    assert store.count() == 3


def test_add_empty_returns_zero(tmp_chroma_dir, embedding_fn) -> None:
    """Adding no chunks returns 0 and leaves the store empty."""
    store = make_store(tmp_chroma_dir, embedding_fn)
    assert store.add_chunks([]) == 0
    assert store.count() == 0


def test_query_returns_most_similar_first(tmp_chroma_dir, embedding_fn) -> None:
    """An exact-text query ranks its own chunk first."""
    store = make_store(tmp_chroma_dir, embedding_fn)
    _seeded(store)
    results = store.query(DOCS[2][1], top_k=3)
    assert len(results) == 3
    assert results[0].doc_id == "northwind"
    assert results[0].score is not None


def test_query_scores_descending(tmp_chroma_dir, embedding_fn) -> None:
    """Query results carry scores in non-increasing order."""
    store = make_store(tmp_chroma_dir, embedding_fn)
    _seeded(store)
    results = store.query("data protection principles", top_k=3)
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_reset_empties(tmp_chroma_dir, embedding_fn) -> None:
    """reset() removes all indexed chunks."""
    store = make_store(tmp_chroma_dir, embedding_fn)
    _seeded(store)
    store.reset()
    assert store.count() == 0
