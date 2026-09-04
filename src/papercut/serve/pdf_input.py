"""Turn a scanned PDF into the text, layout, and visual inputs of a PSS model."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from papercut.data.loaders.hf import HfPssCorpus
from papercut.data.loaders.tabme_pp import (
    extract_layout_from_ocr,
    extract_text_from_ocr,
    extract_visual_from_img,
)
from papercut.streams.types import PageRef, Stream


@dataclass(frozen=True)
class PdfInput:
    """Features extracted from one PDF, ready for boundary prediction."""

    corpus: HfPssCorpus
    stream: Stream


def parse_tesseract_tsv(tsv: str, image_width: int, image_height: int) -> tuple[str, list[float]]:
    """Convert Tesseract word boxes to the same features used for TABME++.

    Tesseract returns pixel coordinates. The training layout extractor expects
    normalised quadrilaterals, so this adapter normalises each box before
    passing it through the shared extraction code.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    lines: list[dict[str, float | str]] = []
    for row in csv.DictReader(tsv.splitlines(), delimiter="\t"):
        word = (row.get("text") or "").strip()
        if not word:
            continue
        try:
            left = float(row["left"]) / image_width
            top = float(row["top"]) / image_height
            width = float(row["width"]) / image_width
            height = float(row["height"]) / image_height
        except (KeyError, TypeError, ValueError):
            continue
        right = min(1.0, left + width)
        bottom = min(1.0, top + height)
        lines.append(
            {
                "Word": word,
                "X1": left,
                "Y1": top,
                "X2": right,
                "Y2": top,
                "X3": right,
                "Y3": bottom,
                "X4": left,
                "Y4": bottom,
            }
        )
    payload = json.dumps({"lines_data": lines})
    return extract_text_from_ocr(payload), extract_layout_from_ocr(payload)


def _binary(path: str | None, name: str) -> str:
    if path:
        return path
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(f"{name} is required. Install it or pass --{name} PATH")
    return resolved


def _render_pages(input_pdf: Path, output_dir: Path, pdftoppm: str, dpi: int) -> list[Path]:
    prefix = output_dir / "page"
    subprocess.run(
        [pdftoppm, "-r", str(dpi), "-png", str(input_pdf), str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    )

    def page_number(path: Path) -> int:
        try:
            return int(path.stem.rsplit("-", maxsplit=1)[1])
        except (IndexError, ValueError) as e:
            raise ValueError(f"Unexpected pdftoppm output name: {path.name}") from e

    pages = sorted(output_dir.glob("page-*.png"), key=page_number)
    if not pages:
        raise ValueError("pdftoppm produced no page images")
    return pages


def _ocr_tsv(image: Path, tesseract: str, languages: str) -> str:
    completed = subprocess.run(
        [tesseract, str(image), "-", "-l", languages, "--psm", "3", "tsv"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def pdf_input(
    input_pdf: Path,
    *,
    languages: str = "eng",
    dpi: int = 200,
    pdftoppm_path: str | None = None,
    tesseract_path: str | None = None,
) -> PdfInput:
    """Render and OCR a PDF without retaining intermediate images on disk."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    pdftoppm = _binary(pdftoppm_path, "pdftoppm")
    tesseract = _binary(tesseract_path, "tesseract")

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("PDF inference needs Pillow") from e

    with tempfile.TemporaryDirectory(prefix="papercut-") as tmp:
        images = _render_pages(input_pdf, Path(tmp), pdftoppm, dpi)
        pages: list[PageRef] = []
        texts: dict[PageRef, str] = {}
        layouts: dict[PageRef, list[float]] = {}
        visuals: dict[PageRef, list[float]] = {}
        source = f"pdf/{input_pdf.name}"
        for index, image_path in enumerate(images):
            with Image.open(image_path) as image:
                text, layout = parse_tesseract_tsv(
                    _ocr_tsv(image_path, tesseract, languages), *image.size
                )
            page = PageRef(source=source, page=index)
            pages.append(page)
            texts[page] = text
            layouts[page] = layout
            visuals[page] = extract_visual_from_img(image_path.read_bytes())

    stream = Stream(pages=tuple(pages))
    corpus = HfPssCorpus(streams=[stream], _texts=texts, _layouts=layouts, _visuals=visuals)
    return PdfInput(corpus=corpus, stream=stream)
