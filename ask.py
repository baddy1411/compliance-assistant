"""Command-line demo client for the compliance assistant.

Example:
    python ask.py "What are the GDPR data-subject rights?" --top-k 5
"""

from __future__ import annotations

import argparse
import sys

from config import get_settings
from rag import ComplianceAssistant


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Ask the compliance assistant a question."
    )
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (1-10, default: 3).",
    )
    parser.add_argument(
        "--chroma-dir",
        default=None,
        help="Override the Chroma persistence directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, ask the question, and print the answer and sources."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 1 <= args.top_k <= 10:
        parser.error("--top-k must be between 1 and 10")

    settings = get_settings()
    if args.chroma_dir:
        settings.chroma_dir = args.chroma_dir

    assistant = ComplianceAssistant(settings=settings)
    result = assistant.ask(args.question, top_k=args.top_k)

    print(result.answer)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(
                f"- [{source['doc_id']}] chunk {source['chunk_id']} "
                f"(score: {source['score']:.4f})"
            )
            excerpt = str(source["excerpt"]).replace("\n", " ").strip()
            print(f"  {excerpt}")
    print(
        f"\n(model: {result.model}, llm_used: {result.llm_used}, "
        f"latency: {result.latency_ms:.1f} ms)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
