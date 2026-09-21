"""Tests for rag.ComplianceAssistant with a fake store and stub LLM.

Uses the real ``ComplianceAssistant`` / ``AskResult`` / ``LLMAnswer``
contracts from the serving layer: sources are dicts with ``doc_id``,
``chunk_id``, ``score`` and ``excerpt``; the stub LLM mimics
``ComplianceLLM`` (``configured`` attribute + ``answer_question(question,
contexts)`` returning ``LLMAnswer``).
"""

from __future__ import annotations

from typing import Any

from conftest import make_chunk

from llm import LLMAnswer
from rag import (
    FALLBACK_PREFIX,
    NO_RESULTS_MESSAGE,
    AskResult,
    ComplianceAssistant,
)
from vectorstore import RetrievedChunk

CANNED = "Canned answer about GDPR principles."


class StubLLM:
    """Test double for ComplianceLLM."""

    def __init__(
        self, answer: str = CANNED, error: Exception | None = None
    ) -> None:
        self.configured = True
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def answer_question(
        self, question: str, contexts: list[dict[str, Any]]
    ) -> LLMAnswer:
        """Record the call, then return the canned answer or raise."""
        self.calls.append((question, contexts))
        if self.error is not None:
            raise self.error
        return LLMAnswer(answer=self.answer, model="stub-model")


class FakeStore:
    """Minimal in-memory stand-in for ComplianceVectorStore."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def query(self, query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the first top_k chunks regardless of the query."""
        return self._chunks[:top_k]

    def count(self) -> int:
        """Return the number of stored chunks."""
        return len(self._chunks)

    def add_chunks(self, chunks) -> int:
        """Add chunks and return the count."""
        self._chunks.extend(chunks)
        return len(chunks)

    def reset(self) -> None:
        """Remove all chunks."""
        self._chunks.clear()


def _retrieved() -> list[RetrievedChunk]:
    chunk = make_chunk(
        doc_id="gdpr_principles",
        index=0,
        text="The GDPR requires lawfulness, fairness and transparency.",
    )
    return [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            score=0.92,
            metadata=chunk.metadata,
        )
    ]


def test_ask_returns_sources_with_scores_and_excerpts() -> None:
    """ask() returns the LLM answer plus scored source excerpts."""
    llm = StubLLM()
    assistant = ComplianceAssistant(store=FakeStore(_retrieved()), llm=llm)
    result = assistant.ask("What are the GDPR principles?")

    assert isinstance(result, AskResult)
    assert CANNED in result.answer
    assert result.llm_used is True
    assert len(llm.calls) == 1
    assert result.sources, "expected at least one source"
    source = result.sources[0]
    assert source["doc_id"] == "gdpr_principles"
    assert isinstance(source["score"], float)
    assert source["excerpt"], "expected a non-empty excerpt"


def test_empty_store_abstains_without_calling_llm() -> None:
    """No retrieved chunks -> abstention message, LLM never called."""
    llm = StubLLM()
    assistant = ComplianceAssistant(store=FakeStore([]), llm=llm)
    result = assistant.ask("What are the French tax deadlines?")

    assert result.answer == NO_RESULTS_MESSAGE
    assert result.llm_used is False
    assert result.sources == []
    assert llm.calls == []


def test_llm_error_falls_back_to_extractive() -> None:
    """A failing LLM yields an extractive fallback answer, not an error."""
    llm = StubLLM(error=RuntimeError("boom"))
    assistant = ComplianceAssistant(store=FakeStore(_retrieved()), llm=llm)
    result = assistant.ask("What are the GDPR principles?")

    assert result.llm_used is False
    assert result.model == "none"
    assert result.answer.startswith(FALLBACK_PREFIX)
    assert "lawfulness, fairness and transparency" in result.answer
    assert result.sources, "fallback keeps the retrieved sources"


def test_low_score_chunks_abstain_without_calling_llm() -> None:
    """Chunks below min_score are treated as irrelevant -> abstention."""
    chunk = make_chunk(
        doc_id="eu_ai_act_high_risk",
        index=0,
        text="Unrelated passage about conformity assessment.",
    )
    low = [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            score=0.10,
            metadata=chunk.metadata,
        )
    ]
    llm = StubLLM()
    assistant = ComplianceAssistant(store=FakeStore(low), llm=llm)
    result = assistant.ask("What are the French tax deadlines?", min_score=0.35)

    assert result.answer == NO_RESULTS_MESSAGE
    assert result.llm_used is False
    assert result.sources == []
    assert llm.calls == []
