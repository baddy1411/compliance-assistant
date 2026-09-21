"""Embedding helpers with lazy model loading.

The SentenceTransformer model is never loaded at import time: it is
instantiated on the first call to :func:`get_embedding_fn` and cached as a
module-level singleton so every subsequent call reuses it.
"""

from __future__ import annotations

from collections.abc import Callable

_model = None
_model_name: str | None = None

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_fn(
    model_name: str | None = None,
) -> Callable[[list[str]], list[list[float]]]:
    """Return a callable mapping ``list[str]`` → ``list[list[float]]``.

    The underlying SentenceTransformer model loads lazily on first use and
    is cached afterwards. Pass ``model_name`` to override the default;
    a different name loads and caches a new model.
    """
    global _model, _model_name
    name = model_name or DEFAULT_MODEL
    if _model is None or _model_name != name:
        from sentence_transformers import SentenceTransformer  # lazy import

        _model = SentenceTransformer(name)
        _model_name = name

    def embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = _model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in vectors]

    return embed


def embed_texts(
    texts: list[str], model_name: str | None = None
) -> list[list[float]]:
    """Embed ``texts`` with the lazily-loaded model, returning dense vectors."""
    return get_embedding_fn(model_name)(texts)
