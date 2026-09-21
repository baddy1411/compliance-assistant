"""Tests for the FastAPI serving layer (api.py).

The real ``app`` is exercised with ``TestClient``; the assistant dependency
is overridden with a stub exposing the same surface the endpoints use:
``ask(question, top_k)``, ``ingest_document(doc_id, title, text, metadata)``
and ``health()``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from api import app, get_assistant_dep
from rag import AskResult


class StubAssistant:
    """Test double for ComplianceAssistant (endpoint-facing surface)."""

    def ask(self, question: str, top_k: int = 3) -> AskResult:
        """Return a canned AskResult."""
        return AskResult(
            answer="stub answer",
            sources=[
                {
                    "doc_id": "gdpr_principles",
                    "chunk_id": "gdpr_principles:0",
                    "score": 0.9,
                    "excerpt": "excerpt text",
                }
            ],
            model="stub-model",
            latency_ms=1.0,
            llm_used=True,
        )

    def ingest_document(
        self,
        doc_id: str,
        title: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Pretend to index a document; return a fixed chunk count."""
        return 7

    def health(self) -> dict[str, Any]:
        """Return a canned health payload."""
        return {
            "status": "ok",
            "chunks_indexed": 7,
            "llm_configured": False,
            "embedding_model": "fake-model",
        }


@pytest.fixture()
def client():
    """TestClient with the assistant dependency stubbed."""
    app.dependency_overrides[get_assistant_dep] = lambda: StubAssistant()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_ask_ok(client) -> None:
    """POST /ask returns the answer shape with sources."""
    response = client.post(
        "/ask", json={"question": "What are the GDPR principles?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "stub answer"
    assert body["llm_used"] is True
    assert body["model"] == "stub-model"
    assert isinstance(body["latency_ms"], float)
    assert body["sources"][0]["doc_id"] == "gdpr_principles"
    assert body["sources"][0]["excerpt"] == "excerpt text"


def test_ask_short_question_422(client) -> None:
    """A 2-character question fails min_length=3 validation."""
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422


def test_ingest_ok(client) -> None:
    """POST /ingest indexes one document and reports chunks added."""
    response = client.post(
        "/ingest",
        json={"doc_id": "d1", "title": "Doc", "text": "hello world"},
    )
    assert response.status_code == 200
    assert response.json() == {"doc_id": "d1", "chunks_added": 7}


def test_health_ok(client) -> None:
    """GET /health returns the service health shape."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] == 7
    assert body["llm_configured"] is False
    assert body["embedding_model"] == "fake-model"
