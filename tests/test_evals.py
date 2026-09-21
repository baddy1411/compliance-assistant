"""Tests for evals.run_evals.retrieval_metrics (pure function, no I/O)."""

from __future__ import annotations

import pytest

from evals.run_evals import retrieval_metrics


def test_perfect_ranking() -> None:
    """Every expected doc ranked first -> recall 1.0, MRR 1.0."""
    ranked = [["a", "b"], ["c"]]
    expected = ["a", "c"]
    assert retrieval_metrics(ranked, expected, k=2) == {
        "recall@k": 1.0,
        "mrr": 1.0,
    }


def test_imperfect_ranking() -> None:
    """Hand-computed case: recall misses, MRR counts the deep hit.

    ranked[0] = ["x", "y", "a"], expected "a", k=2 -> "a" is at rank 3,
    outside top-2 -> recall miss; reciprocal rank 1/3.
    ranked[1] = ["b", "z"], expected "c" -> absent -> miss, RR 0.
    recall@2 = 0/2 = 0.0; MRR = (1/3 + 0) / 2 = 1/6.
    """
    ranked = [["x", "y", "a"], ["b", "z"]]
    expected = ["a", "c"]
    metrics = retrieval_metrics(ranked, expected, k=2)
    assert metrics["recall@k"] == pytest.approx(0.0)
    assert metrics["mrr"] == pytest.approx(1 / 6)


def test_k_truncation_applies_to_recall_only() -> None:
    """recall@k uses the top-k slice; MRR uses the full ranking.

    "a" at rank 2 with k=1 -> recall miss, but MRR = 1/2.
    """
    metrics = retrieval_metrics([["x", "a"]], ["a"], k=1)
    assert metrics["recall@k"] == pytest.approx(0.0)
    assert metrics["mrr"] == pytest.approx(0.5)


def test_none_expected_items_skipped() -> None:
    """Abstention items (expected None) contribute to neither metric."""
    assert retrieval_metrics([["a"], ["b"]], [None, None], k=2) == {
        "recall@k": 0.0,
        "mrr": 0.0,
    }


def test_mixed_none_and_scored_items() -> None:
    """None items are skipped while scored items still count."""
    metrics = retrieval_metrics([["a", "b"], ["zzz"]], [None, "zzz"], k=2)
    assert metrics["recall@k"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
