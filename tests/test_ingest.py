"""Tests for ingest.load_document.

Assumes ``load_document(path)`` returns a Document whose ``doc_id`` is the
file stem, raises ``ValueError`` for unsupported suffixes and
``FileNotFoundError`` for missing files. See NOTES_OPS.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingest import load_document

BUILD_ROOT = Path(__file__).resolve().parent.parent


def test_load_markdown(tmp_path: Path) -> None:
    """A .md file loads with its text and stem as doc_id."""
    path = tmp_path / "gdpr_principles.md"
    path.write_text("# Principles\n\nLawfulness, fairness and transparency.")
    doc = load_document(str(path))
    assert doc.doc_id == "gdpr_principles"
    assert "Lawfulness, fairness and transparency." in doc.text


def test_load_txt(tmp_path: Path) -> None:
    """A .txt file loads with its text and stem as doc_id."""
    path = tmp_path / "notes.txt"
    path.write_text("Retention period is 24 months.")
    doc = load_document(str(path))
    assert doc.doc_id == "notes"
    assert "Retention period is 24 months." in doc.text


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    """Unknown suffixes raise ValueError."""
    path = tmp_path / "data.xyz"
    path.write_text("not a document")
    with pytest.raises(ValueError):
        load_document(str(path))


def test_missing_file_raises() -> None:
    """Missing files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_document(str(BUILD_ROOT / "data" / "does_not_exist.md"))


def test_load_pdf_conditional() -> None:
    """PDF loading is tested only when pypdf and the sample PDF exist."""
    try:
        import pypdf  # noqa: F401
    except ImportError:
        pytest.skip("pypdf not installed")
    pdf = BUILD_ROOT / "data" / "company_privacy_policy.pdf"
    if not pdf.exists():
        pytest.skip("data/company_privacy_policy.pdf missing")
    doc = load_document(str(pdf))
    assert doc.doc_id == "company_privacy_policy"
    assert doc.text.strip()
