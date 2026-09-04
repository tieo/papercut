"""Write clean per-document PDFs from page-boundary and blank-page decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader, PdfWriter


@dataclass(frozen=True)
class SplitOutput:
    """One written document and its zero-indexed source pages."""

    path: Path
    source_page_indices: tuple[int, ...]


def document_page_indices(
    boundaries: Sequence[bool], blank_pages: Sequence[bool]
) -> tuple[tuple[int, ...], ...]:
    """Group nonblank source pages into output documents.

    A boundary starts a new output document even when it lands on a blank
    page. Blank pages are omitted, and blank-only groups produce no file.
    This keeps duplex scanner backsides out of the output while preserving
    document boundaries around them.
    """
    if not boundaries:
        raise ValueError("Need at least one page")
    if len(boundaries) != len(blank_pages):
        raise ValueError("boundaries and blank_pages must have equal length")
    if not boundaries[0]:
        raise ValueError("The first page must start a document")

    documents: list[tuple[int, ...]] = []
    current: list[int] = []
    for index, (starts_document, is_blank) in enumerate(zip(boundaries, blank_pages, strict=True)):
        if starts_document and current:
            documents.append(tuple(current))
            current = []
        if not is_blank:
            current.append(index)
    if current:
        documents.append(tuple(current))
    return tuple(documents)


def write_split_pdfs(
    input_pdf: Path,
    output_dir: Path,
    boundaries: Sequence[bool],
    blank_pages: Sequence[bool],
    filename_prefix: str = "document",
) -> tuple[SplitOutput, ...]:
    """Write one PDF per predicted document, excluding confirmed blank pages.

    The input file is only read. Outputs are named ``<prefix>_001.pdf``,
    ``<prefix>_002.pdf``, and so on in ``output_dir``.
    """
    reader = PdfReader(str(input_pdf))
    if len(reader.pages) != len(boundaries):
        raise ValueError(
            f"PDF has {len(reader.pages)} pages but received {len(boundaries)} boundaries"
        )
    if not filename_prefix or Path(filename_prefix).name != filename_prefix:
        raise ValueError("filename_prefix must be a non-empty filename stem")

    groups = document_page_indices(boundaries, blank_pages)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[SplitOutput] = []
    for number, indices in enumerate(groups, start=1):
        path = output_dir / f"{filename_prefix}_{number:03d}.pdf"
        writer = PdfWriter()
        for index in indices:
            writer.add_page(reader.pages[index])
        with path.open("wb") as f:
            writer.write(f)
        outputs.append(SplitOutput(path=path, source_page_indices=indices))
    return tuple(outputs)
