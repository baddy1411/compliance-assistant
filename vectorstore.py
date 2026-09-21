"""Chroma-backed vector store for compliance document chunks.

The Chroma client is created per :class:`ComplianceVectorStore` instance
(never at import time). The embedding function is injectable so unit tests
can pass a deterministic fake instead of loading a real model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chromadb

from ingest import Chunk


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity query.

    ``score`` is cosine similarity in [0, 1]; higher means more relevant.
    """

    doc_id: str
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class ComplianceVectorStore:
    """Persistent Chroma vector store for compliance document chunks."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "compliance_docs",
        embedding_fn=None,
    ) -> None:
        """Create (or open) the store.

        Args:
            persist_dir: Directory Chroma persists to.
            collection_name: Name of the Chroma collection.
            embedding_fn: Callable ``list[str] -> list[list[float]]``.
                Defaults to :func:`embeddings.get_embedding_fn` (lazy model).
        """
        if embedding_fn is None:
            from embeddings import get_embedding_fn

            embedding_fn = get_embedding_fn()
        self.embedding_fn = embedding_fn
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #
    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the collection; returns the number added.

        An empty list is a no-op returning 0. Metadata is flattened to
        Chroma-safe scalar values (str/int/float/bool).
        """
        if not chunks:
            return 0
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        embeddings = self.embedding_fn(documents)
        metadatas = [self._flatten_metadata(c) for c in chunks]
        self._collection.add(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )
        return len(chunks)

    @staticmethod
    def _flatten_metadata(chunk: Chunk) -> dict:
        meta = dict(chunk.metadata or {})
        flat = {
            "doc_id": chunk.doc_id,
            "title": str(meta.get("title", "")),
            "source_path": str(meta.get("source_path", "")),
        }
        chunk_index = meta.get("chunk_index", 0)
        try:
            flat["chunk_index"] = int(chunk_index)
        except (TypeError, ValueError):
            flat["chunk_index"] = 0
        for key, value in meta.items():
            if key in flat or not isinstance(value, (str, int, float, bool)):
                continue
            flat[str(key)] = value
        return flat

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    def query(self, text: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, best-first.

        Chroma cosine distance ``d`` is converted to a similarity score
        ``max(0.0, 1.0 - d)`` in [0, 1].
        """
        if top_k <= 0:
            return []
        query_embedding = self.embedding_fn([text])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        retrieved: list[RetrievedChunk] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = dict(metadata or {})
            retrieved.append(
                RetrievedChunk(
                    doc_id=str(metadata.get("doc_id", "")),
                    chunk_id=str(chunk_id),
                    text=str(document or ""),
                    metadata=metadata,
                    score=max(0.0, 1.0 - float(distance)),
                )
            )
        # results arrive ordered by distance ascending; ensure best-first.
        retrieved.sort(key=lambda r: r.score, reverse=True)
        return retrieved

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (destructive)."""
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
