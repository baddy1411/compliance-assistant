"""LLM answering layer for the compliance assistant.

Wraps an OpenAI-compatible chat-completions API. The ``openai`` package is
imported lazily inside the answering method so this module (and the rest of
the serving layer) can be imported without it installed; it is only required
when an answer is actually generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from config import Settings


@dataclass
class LLMAnswer:
    """Result of a single LLM answering call."""

    answer: str
    model: str


class ComplianceLLM:
    """Answers compliance questions with an OpenAI-compatible LLM.

    The model is instructed to answer strictly from the retrieved context and
    to cite every factual claim with its source document id.
    """

    SYSTEM_PROMPT = (
        "You are a compliance assistant. Answer the user's question using ONLY "
        "the context provided below. Cite every factual claim with the source "
        "document id in square brackets, e.g. [gdpr_principles]. Do not use any "
        "outside knowledge and do not speculate. If the provided context does "
        "not contain the answer, reply with exactly: \"I don't have enough "
        "information in the indexed documents to answer that.\" Keep your "
        "answer concise."
    )

    def __init__(self, settings: Settings) -> None:
        """Store connection details from the application settings."""
        self.base_url: str | None = settings.llm_base_url
        self.api_key: str | None = settings.llm_api_key
        self.model: str = settings.llm_model

    @property
    def configured(self) -> bool:
        """Return True when an API key is set and non-empty."""
        return bool(self.api_key and str(self.api_key).strip())

    def answer_question(
        self, question: str, contexts: list[dict[str, Any]]
    ) -> LLMAnswer:
        """Generate an answer grounded in the given retrieved contexts.

        Args:
            question: The user's question.
            contexts: Retrieved passages, each a dict with ``doc_id`` and
                ``text`` keys.

        Returns:
            An :class:`LLMAnswer` with the generated text and model name.

        Raises:
            RuntimeError: If the LLM is not configured (no API key), if the
                ``openai`` package is missing, or if the API call fails.
        """
        if not self.configured:
            raise RuntimeError(
                "LLM is not configured: no API key was provided. "
                "Set the LLM_API_KEY environment variable "
                "(Settings.llm_api_key) to enable LLM-generated answers."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for LLM answers but is not "
                "installed. Install it with: pip install openai"
            ) from exc

        context_block = "\n".join(
            f"[{context['doc_id']}] {context['text']}" for context in contexts
        )
        user_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer with citations."
        )

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = (response.choices[0].message.content or "").strip()
        return LLMAnswer(answer=content, model=self.model)
