from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from papercut.serve.split_pdf import document_page_indices, write_split_pdfs


def _pdf(path: Path, pages: int) -> Path:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as f:
        writer.write(f)
    return path


def test_document_page_indices_removes_blanks_without_empty_documents() -> None:
    assert document_page_indices(
        boundaries=(True, False, False, True, False),
        blank_pages=(False, True, False, False, True),
    ) == ((0, 2), (3,))


def test_document_page_indices_uses_boundary_after_blank_separator() -> None:
    assert document_page_indices(
        boundaries=(True, False, True),
        blank_pages=(False, True, False),
    ) == ((0,), (2,))


def test_document_page_indices_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="equal length"):
        document_page_indices((True,), ())
    with pytest.raises(ValueError, match="first page"):
        document_page_indices((False,), (False,))


def test_write_split_pdfs_excludes_blank_pages(tmp_path: Path) -> None:
    source = _pdf(tmp_path / "input.pdf", pages=5)
    outputs = write_split_pdfs(
        source,
        tmp_path / "output",
        boundaries=(True, False, False, True, False),
        blank_pages=(False, True, False, False, True),
    )

    assert [output.source_page_indices for output in outputs] == [(0, 2), (3,)]
    assert [len(PdfReader(str(output.path)).pages) for output in outputs] == [2, 1]
    assert len(PdfReader(str(source)).pages) == 5
