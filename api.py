"""FastAPI serving layer for the compliance assistant.

Exposes ``/ask``, ``/ingest`` and ``/health`` endpoints. The assistant is
resolved through :func:`get_assistant_dep` so tests can substitute a fake via
``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

import rag
from rag import ComplianceAssistant

app = FastAPI(title="compliance-assistant", version="0.1.0")


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(min_length=3)
    top_k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    """A retrieved source passage backing an answer."""

    doc_id: str
    chunk_id: str
    score: float
    excerpt: str


class AskResponse(BaseModel):
    """Response body for POST /ask."""

    answer: str
    sources: list[Source]
    model: str
    latency_ms: float
    llm_used: bool


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    doc_id: str
    title: str
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    doc_id: str
    chunks_added: int


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    chunks_indexed: int
    llm_configured: bool
    embedding_model: str


def get_assistant_dep() -> ComplianceAssistant:
    """FastAPI dependency returning the shared assistant instance."""
    return rag.get_assistant()


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(
    body: AskRequest,
    assistant: Annotated[ComplianceAssistant, Depends(get_assistant_dep)],
) -> AskResponse:
    """Answer a compliance question with retrieval-augmented generation."""
    try:
        result = assistant.ask(body.question, top_k=body.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(
        answer=result.answer,
        sources=[Source(**source) for source in result.sources],
        model=result.model,
        latency_ms=result.latency_ms,
        llm_used=result.llm_used,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(
    body: IngestRequest,
    assistant: Annotated[ComplianceAssistant, Depends(get_assistant_dep)],
) -> IngestResponse:
    """Chunk and index a single document supplied in the request body."""
    try:
        chunks_added = assistant.ingest_document(
            body.doc_id, body.title, body.text, body.metadata
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestResponse(doc_id=body.doc_id, chunks_added=chunks_added)


@app.get("/health", response_model=HealthResponse)
def health_endpoint(
    assistant: Annotated[ComplianceAssistant, Depends(get_assistant_dep)],
) -> HealthResponse:
    """Report service health and index statistics."""
    return HealthResponse(**assistant.health())


@app.get("/")
def root() -> dict[str, str]:
    """Service landing payload pointing at the interactive docs."""
    return {"service": "compliance-assistant", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000)
