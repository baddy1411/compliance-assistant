"""Offline-capable eval harness for the compliance RAG assistant.

Runs the questions in ``eval_set.json`` against a ``ComplianceAssistant``
backed by a TEMPORARY Chroma directory (``tempfile.mkdtemp`` -- never the
default one), then reports:

* ``recall@k`` -- fraction of questions whose expected document appears in
  the top-k retrieved doc ids. Items with ``expected_doc_id: null`` are
  skipped here; they exercise the abstention path instead.
* ``mrr`` -- mean reciprocal rank of the expected doc over the full ranked
  list (null-expected items skipped).
* ``faithfulness`` -- for each answered question, an LLM-as-judge check that
  every factual claim in the answer is supported by the retrieved context.
  When no LLM is configured this is reported as
  ``"skipped (no LLM_API_KEY)"``.
* ``abstention`` -- the out-of-scope question must be refused with a
  "couldn't find" / "don't have enough information" style message.

Assumes the ``data/`` directory holds documents whose ``doc_id`` values match
the ``expected_doc_id`` values in ``eval_set.json`` (``gdpr_principles``,
``eu_ai_act_high_risk``, ``company_privacy_policy``).

Usage:
    python evals/run_evals.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from rag import ComplianceAssistant
from vectorstore import ComplianceVectorStore

EVALS_DIR = Path(__file__).resolve().parent
BUILD_ROOT = EVALS_DIR.parent
LAST_RUN = EVALS_DIR / "last_run.json"

#: Substrings (lowercased) that mark a proper abstention answer.
ABSTAIN_MARKERS = ("couldn't find", "don't have enough information")

#: Judge instruction sent as the "question"; retrieved passages are passed as
#: the LLM contexts argument, matching ComplianceLLM.answer_question's
#: (question, contexts) signature.
JUDGE_INSTRUCTION = (
    "Given the context passages above, is every factual claim in the answer "
    "below supported by the context? Reply YES or NO, then one sentence "
    "explaining.\n\nAnswer:\n{answer}"
)


def retrieval_metrics(
    ranked: list[list[str]], expected: list[str | None], k: int
) -> dict[str, float]:
    """Compute recall@k and MRR over ranked doc-id lists.

    Items whose expected value is None are skipped (they exercise the
    abstention path, measured separately). ``recall@k`` uses the top-k
    slice of each ranking; MRR uses the full ranked list.

    Args:
        ranked: Per-question ranked lists of retrieved doc ids.
        expected: Per-question expected doc id (None = abstention item).
        k: Cutoff for recall.

    Returns:
        ``{"recall@k": float, "mrr": float}``; both 0.0 when there is no
        scorable item.
    """
    hits = 0
    reciprocal_sum = 0.0
    n = 0
    for ranking, want in zip(ranked, expected):
        if want is None:
            continue
        n += 1
        if want in ranking[:k]:
            hits += 1
        if want in ranking:
            reciprocal_sum += 1.0 / (ranking.index(want) + 1)
    if n == 0:
        return {"recall@k": 0.0, "mrr": 0.0}
    return {"recall@k": hits / n, "mrr": reciprocal_sum / n}


def _doc_id(source: Any) -> str | None:
    """Extract a doc id from a source dict (or duck-typed object)."""
    if isinstance(source, dict):
        return source.get("doc_id")
    return getattr(source, "doc_id", None)


def _excerpt(source: Any, limit: int = 800) -> str:
    """Extract the excerpt text from a source dict (or duck-typed object)."""
    if isinstance(source, dict):
        text = source.get("excerpt") or source.get("text") or ""
    else:
        text = getattr(source, "excerpt", None) or getattr(source, "text", "") or ""
    return str(text)[:limit]


def _parse_judge_verdict(text: str) -> bool | None:
    """Parse a YES/NO verdict from the judge's reply; None if unparseable."""
    cleaned = text.strip().upper()
    if cleaned.startswith("YES"):
        return True
    if cleaned.startswith("NO"):
        return False
    return None


def _judge_faithfulness(llm: Any, sources: list[Any], answer: str) -> bool | None:
    """Ask the LLM judge whether the answer is supported by the sources."""
    contexts = [
        {"doc_id": _doc_id(source) or "?", "text": _excerpt(source)}
        for source in sources
    ]
    response = llm.answer_question(JUDGE_INSTRUCTION.format(answer=answer), contexts)
    verdict_text = getattr(response, "answer", None) or str(response)
    return _parse_judge_verdict(str(verdict_text))


def run(top_k: int = 5, chroma_dir: str | None = None) -> dict[str, Any]:
    """Run the eval set and return the metrics dict (also printed + saved).

    Args:
        top_k: Retrieval cutoff used for asking and for recall@k.
        chroma_dir: Optional pre-created dir; otherwise a fresh temporary
            directory is used (never the default ``chroma_dir``).

    Returns:
        Dict with recall@k, MRR, faithfulness, abstention result and
        per-item details. Also written to ``evals/last_run.json``.
    """
    with open(EVALS_DIR / "eval_set.json", encoding="utf-8") as handle:
        items = json.load(handle)

    tmp_dir = chroma_dir or tempfile.mkdtemp(prefix="compliance-eval-")
    settings = get_settings()
    # Point at the temporary dir without mutating any shared settings object.
    if hasattr(settings, "model_copy"):
        settings = settings.model_copy(update={"chroma_dir": tmp_dir})
    else:  # pragma: no cover - non-pydantic Settings fallback
        settings.chroma_dir = tmp_dir

    # Mirror rag.py's store construction so evals use the real indexing path.
    store = ComplianceVectorStore(persist_dir=tmp_dir)
    assistant = ComplianceAssistant(settings=settings, store=store)
    data_dir = BUILD_ROOT / "data"
    chunks_ingested = 0
    if data_dir.is_dir():
        # Assistant delegates to ingest.ingest_folder(folder, store).
        chunks_ingested = assistant.ingest_folder(str(data_dir))
    else:
        print(f"WARNING: data dir {data_dir} missing; evals run on an empty store.")

    llm = assistant.llm
    llm_ready = bool(getattr(llm, "configured", False))

    ranked: list[list[str]] = []
    expected: list[str | None] = []
    per_item: list[dict[str, Any]] = []
    faithful_votes: list[bool] = []
    abstention_pass: bool | None = None

    for item in items:
        result = assistant.ask(item["question"], top_k=top_k)
        sources = result.sources or []
        doc_ids = [d for d in (_doc_id(source) for source in sources) if d]
        ranked.append(doc_ids)
        expected.append(item["expected_doc_id"])

        want = item["expected_doc_id"]
        rank = doc_ids.index(want) + 1 if want in doc_ids else None
        hit = want is not None and want in doc_ids[:top_k]

        faithful: bool | None = None
        if want is not None and llm_ready:
            faithful = _judge_faithfulness(llm, sources[:top_k], result.answer)
            if faithful is not None:
                faithful_votes.append(faithful)

        if want is None:
            answer_low = (result.answer or "").lower()
            abstention_pass = any(
                marker in answer_low for marker in ABSTAIN_MARKERS
            )

        per_item.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_doc_id": want,
                "top1_doc_id": doc_ids[0] if doc_ids else None,
                "rank": rank,
                f"hit@{top_k}": hit,
                "faithful": faithful,
            }
        )

    metrics = retrieval_metrics(ranked, expected, top_k)
    if faithful_votes:
        faithfulness: Any = sum(faithful_votes) / len(faithful_votes)
    elif llm_ready:
        faithfulness = "no judgeable answers"
    else:
        faithfulness = "skipped (no LLM_API_KEY)"

    report: dict[str, Any] = {
        "k": top_k,
        "n_items": len(items),
        "chunks_ingested": chunks_ingested,
        "recall@k": metrics["recall@k"],
        "mrr": metrics["mrr"],
        "faithfulness": faithfulness,
        "abstention_pass": abstention_pass,
        "per_item": per_item,
    }

    print(
        f"\nCompliance RAG evals "
        f"(top_k={top_k}, chunks={chunks_ingested}, tmp chroma: {tmp_dir})"
    )
    print(f"{'id':<10}{'expected':<24}{'top1':<24}{'rank':<6}{'hit@k':<7}faithful")
    for row in per_item:
        print(
            f"{row['id']:<10}{row['expected_doc_id']!s:<24}"
            f"{row['top1_doc_id']!s:<24}{row['rank']!s:<6}"
            f"{row[f'hit@{top_k}']!s:<7}{row['faithful']}"
        )
    print(f"\nrecall@{top_k}: {metrics['recall@k']:.3f}")
    print(f"MRR:          {metrics['mrr']:.3f}")
    if isinstance(faithfulness, str):
        print(f"faithfulness: {faithfulness}")
    else:
        print(f"faithfulness: {faithfulness:.3f}")
    print(f"abstention (out-of-scope refused): {abstention_pass}")

    with open(LAST_RUN, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {LAST_RUN}")
    return report


if __name__ == "__main__":
    run()
