"""Document loading and chunking for the compliance RAG data layer.

Chunking choice
---------------
Compliance documents are ingested with a recursive character splitter at
~500 characters per chunk (≈ 100–130 tokens for English legal text). That
size keeps each chunk focused on a single provision or obligation, which
improves retrieval precision: an embedding of one GDPR article ranks more
reliably than an embedding of a whole multi-article page. A 50-character
overlap preserves sentence continuity across chunk boundaries so a clause
that starts at the end of one chunk still carries enough context to match
in the next. The splitter prefers paragraph and sentence boundaries
("\\n\\n" → "\\n" → ". " → ...) so splits never land mid-sentence unless
the text forces it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from config import get_settings

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Document:
    """A raw document loaded from disk."""

    doc_id: str
    title: str
    text: str
    source_path: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A text chunk ready for embedding and storage.

    ``metadata`` always contains ``title``, ``source_path``, and
    ``chunk_index`` so retrieval results can cite their origin.
    """

    doc_id: str
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF page by page, joined with blank lines."""
    from pypdf import PdfReader  # lazy: pypdf only needed for .pdf input

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p).strip()


def load_document(path: str | Path) -> Document:
    """Load a .md/.txt/.pdf file into a :class:`Document`.

    ``doc_id`` is the file stem (name without extension).

    Raises:
        FileNotFoundError: If ``path`` does not exist or is not a file.
        ValueError: If the suffix is unsupported or no text could be extracted.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"document not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        text = _extract_pdf_text(p)
    elif suffix in {".md", ".txt"}:
        text = p.read_text(encoding="utf-8").strip()
    else:
        raise ValueError(
            f"unsupported file type {suffix!r} for {p.name}; "
            f"supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if not text:
        raise ValueError(f"no text extracted from {p}")

    return Document(
        doc_id=p.stem,
        title=p.stem,
        text=text,
        source_path=str(p),
        metadata={"suffix": suffix},
    )


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursively split ``text`` into non-empty chunks of ≤ ``chunk_size``.

    The first separator that occurs in the current piece is used, preferring
    paragraph and sentence boundaries. The empty-string separator is the
    final fallback and splits on raw characters. If ``text`` is shorter than
    ``chunk_size`` it is returned as a single chunk. ``chunk_overlap`` is
    applied as a sliding window on the final merge so context bleeds across
    boundaries.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split_once(piece: str) -> list[str]:
        for sep in SEPARATORS:
            if sep == "":
                break
            if sep in piece:
                return [p.strip() for p in piece.split(sep)]
        return [piece]

    # Recursively split until every piece fits.
    work = [text]
    done: list[str] = []
    while work:
        piece = work.pop(0)
        if len(piece) <= chunk_size:
            done.append(piece)
        else:
            parts = split_once(piece)
            if len(parts) == 1 and parts[0] == piece:
                # No separator found; hard-split on characters.
                for i in range(0, len(piece), chunk_size):
                    done.append(piece[i : i + chunk_size])
            else:
                work = parts + work

    # Merge short fragments back up to chunk_size with overlap.
    merged: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in done:
        piece = piece.strip()
        if not piece:
            continue
        addition = len(piece) + (1 if current else 0)
        if current and current_len + addition > chunk_size:
            merged.append(" ".join(current))
            # Start the next chunk with the overlap tail of this one.
            tail = " ".join(current)
            if chunk_overlap > 0:
                tail = tail[-chunk_overlap:].strip()
                current = [tail] if tail else []
                current_len = len(tail)
            else:
                current, current_len = [], 0
        current.append(piece)
        current_len += addition
    if current:
        merged.append(" ".join(current))

    return [c for c in merged if c.strip()]


def chunk_document(
    doc: Document, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[Chunk]:
    """Split ``doc`` into :class:`Chunk` objects.

    ``chunk_id`` is ``f"{doc_id}:::{i:04d}"``. Never returns empty chunks.
    """
    texts = _split_text(doc.text, chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for i, text in enumerate(texts):
        chunks.append(
            Chunk(
                doc_id=doc.doc_id,
                chunk_id=f"{doc.doc_id}:::{i:04d}",
                text=text,
                metadata={
                    "title": doc.title,
                    "source_path": doc.source_path,
                    "chunk_index": i,
                },
            )
        )
    return chunks


def ingest_folder(
    folder: str | Path,
    store,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> int:
    """Load, chunk, and store every .md/.txt/.pdf in ``folder`` (sorted).

    Unreadable files are skipped with a printed warning instead of
    crashing the run. Returns the total number of chunks added.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"ingest folder not found: {folder_path}")

    total = 0
    files = sorted(
        p
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    for path in files:
        try:
            doc = load_document(path)
        except (
            FileNotFoundError,
            ValueError,
            OSError,
            UnicodeDecodeError,
            ModuleNotFoundError,  # e.g. pypdf missing while a PDF is present
        ) as exc:
            print(f"[ingest] WARNING: skipping {path.name}: {exc}")
            continue
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        total += store.add_chunks(chunks)
        print(f"[ingest] {path.name}: {len(chunks)} chunks")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest .md/.txt/.pdf documents into the Chroma vector store."
    )
    parser.add_argument("--folder", required=True, help="folder of documents to ingest")
    parser.add_argument(
        "--chroma-dir", default=None, help="Chroma persistence dir (env CHROMA_DIR)"
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    from vectorstore import ComplianceVectorStore

    settings = get_settings()
    chroma_dir = args.chroma_dir or settings.chroma_dir
    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = args.chunk_overlap or settings.chunk_overlap

    store = ComplianceVectorStore(persist_dir=chroma_dir)
    total = ingest_folder(args.folder, store, chunk_size, chunk_overlap)
    print(f"[ingest] done: {total} chunks in collection '{settings.collection_name}'")


if __name__ == "__main__":
    main()
