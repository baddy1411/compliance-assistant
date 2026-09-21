"""Retrieval-augmented generation orchestration for the compliance assistant.

This module wires together the sibling layers (settings, ingestion,
embeddings, vector store) with the LLM answering layer defined in
:mod:`llm`. All collaborators are injectable so the assistant can be tested
without a real vector store or LLM.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import ingest
from config import Settings, get_settings
from embeddings import get_embedding_fn
from ingest import Chunk, Document, chunk_document
from llm import ComplianceLLM
from vectorstore import ComplianceVectorStore, RetrievedChunk

#: Message returned when retrieval finds nothing relevant.
NO_RESULTS_MESSAGE = (
    "I couldn't find relevant information in the indexed compliance documents."
)

#: Minimum cosine-similarity score for a retrieved chunk to count as
#: relevant. Calibrated on the sample corpus with all-MiniLM-L6-v2:
#: in-scope questions score >= 0.47, off-topic questions <= 0.25.
DEFAULT_MIN_SCORE = 0.35

#: Prefix for the extractive fallback answer used when the LLM is unavailable.
FALLBACK_PREFIX = "LLM unavailable — showing most relevant passages:"


@dataclass
class AskResult:
    """Result of asking the assistant a question."""

    answer: str
    sources: list[dict[str, Any]]
    model: str
    latency_ms: float
    llm_used: bool


class ComplianceAssistant:
    """High-level RAG assistant over the compliance document index."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: ComplianceVectorStore | None = None,
        llm: ComplianceLLM | None = None,
    ) -> None:
        """Create the assistant, building real collaborators by default.

        Args:
            settings: Application settings; loaded via ``get_settings()``
                when omitted.
            store: Vector store to use; a real
                :class:`ComplianceVectorStore` backed by
                ``settings.chroma_dir`` is built when omitted.
            llm: LLM answerer to use; a real :class:`ComplianceLLM` is built
                when omitted.
        """
        self.settings: Settings = settings or get_settings()
        self.store: ComplianceVectorStore = (
            store
            if store is not None
            else ComplianceVectorStore(persist_dir=self.settings.chroma_dir)
        )
        self.llm: ComplianceLLM = (
            llm if llm is not None else ComplianceLLM(self.settings)
        )
        self._embedding_fn: Callable[..., Any] | None = None
        self._embedding_lock = threading.Lock()

    @property
    def embedding_fn(self) -> Callable[..., Any]:
        """Lazily resolve the embedding function from the embeddings module.

        Kept available for callers that need to embed text the same way the
        index was built; resolution is deferred so constructing the assistant
        never forces an embedding model load.
        """
        if self._embedding_fn is None:
            with self._embedding_lock:
                if self._embedding_fn is None:
                    self._embedding_fn = get_embedding_fn()
        return self._embedding_fn

    def ask(
        self, question: str, top_k: int = 3, min_score: float = DEFAULT_MIN_SCORE
    ) -> AskResult:
        """Answer a question using retrieval plus the LLM, with fallback.

        Retrieval is attempted first; when no chunks are found — or the best
        chunk scores below ``min_score`` (off-topic question) — a fixed
        message is returned without calling the LLM. When the LLM is not
        configured or its API call fails, an extractive fallback answer made
        of the top retrieved passages is returned instead.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            min_score: Minimum cosine-similarity score for the top chunk to
                count as relevant; below this the assistant abstains.

        Returns:
            An :class:`AskResult` with the answer, sources, model name,
            latency in milliseconds, and whether the LLM was used.
        """
        started = time.perf_counter()
        try:
            retrieved: list[RetrievedChunk] = self.store.query(
                question, top_k=top_k
            )
        except Exception as exc:
            raise RuntimeError(f"Vector-store retrieval failed: {exc}") from exc

        # Drop chunks below the relevance bar; an off-topic question may
        # still return nearest neighbours that answer nothing.
        retrieved = [chunk for chunk in retrieved if chunk.score >= min_score]

        if not retrieved:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return AskResult(
                answer=NO_RESULTS_MESSAGE,
                sources=[],
                model="none",
                latency_ms=latency_ms,
                llm_used=False,
            )

        sources = [
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "score": float(chunk.score),
                "excerpt": chunk.text[:200],
            }
            for chunk in retrieved
        ]
        contexts = [
            {"doc_id": chunk.doc_id, "text": chunk.text} for chunk in retrieved
        ]

        try:
            llm_answer = self.llm.answer_question(question, contexts)
        except RuntimeError:
            # LLM not configured or API failure: fall back to extractive
            # passages so the user still gets the relevant content.
            passages = "\n\n".join(
                f"[{chunk.doc_id}] {chunk.text}" for chunk in retrieved
            )
            answer = f"{FALLBACK_PREFIX}\n\n{passages}"
            model, llm_used = "none", False
        else:
            answer, model, llm_used = (
                llm_answer.answer,
                llm_answer.model,
                True,
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        return AskResult(
            answer=answer,
            sources=sources,
            model=model,
            latency_ms=latency_ms,
            llm_used=llm_used,
        )

    def ingest_document(
        self,
        doc_id: str,
        title: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Chunk a document and add it to the vector store.

        Args:
            doc_id: Unique document identifier.
            title: Human-readable document title.
            text: Full document text.
            metadata: Optional extra metadata merged over ``{"title": title}``.

        Returns:
            The number of chunks added to the store.
        """
        merged_metadata: dict[str, Any] = {"title": title}
        if metadata:
            merged_metadata.update(metadata)
        # ingest.Document is the sibling's canonical document type; synthetic
        # documents (from the API or Kafka) carry an empty source_path.
        document = Document(
            doc_id=doc_id,
            title=title,
            text=text,
            source_path="",
            metadata=merged_metadata,
        )
        chunks: list[Chunk] = chunk_document(document)
        return self.store.add_chunks(chunks)

    def ingest_folder(self, folder: str) -> int:
        """Index every supported document found in ``folder``.

        Delegates to :func:`ingest.ingest_folder` with this assistant's store.

        Returns:
            The number of chunks added to the store.
        """
        return ingest.ingest_folder(folder, self.store)

    def health(self) -> dict[str, Any]:
        """Return a health summary of the assistant and its dependencies."""
        return {
            "status": "ok",
            "chunks_indexed": self.store.count(),
            "llm_configured": self.llm.configured,
            "embedding_model": self.settings.embedding_model,
        }


_assistant: ComplianceAssistant | None = None
_assistant_lock = threading.Lock()


def get_assistant() -> ComplianceAssistant:
    """Return the module-level assistant singleton, creating it on first use."""
    global _assistant
    if _assistant is None:
        with _assistant_lock:
            if _assistant is None:
                _assistant = ComplianceAssistant()
    return _assistant
