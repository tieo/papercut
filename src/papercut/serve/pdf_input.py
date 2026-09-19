"""Turn a scanned PDF into the text, layout, and visual inputs of a PSS model."""

from __future__ import annotations

import csv
import io
import json
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from papercut.data.loaders.hf import HfPssCorpus
from papercut.data.loaders.tabme_pp import (
    extract_layout_from_ocr,
    extract_text_from_ocr,
    extract_visual_from_img,
)
from papercut.streams.types import PageRef, Stream

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


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
    # Tesseract writes plain tab separated columns and quotes nothing, so a
    # recognised quote character makes the csv module treat everything up to
    # the next one as a single field, and a page's coordinates end up inside
    # its text. QUOTE_NONE keeps every column where it belongs.
    for row in csv.DictReader(tsv.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE):
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


_SPARSE_PAGE_CHARS = 200

_ROTATION = re.compile(r"^Rotate:\s*(\d+)", re.MULTILINE)


def detect_rotation(image: Path, tesseract: str) -> int:
    """Degrees the page must turn clockwise to stand upright, 0 if unknown.

    A feeder takes pages in whatever way they were put in, and a duplex pass
    turns every backside upside down. Tesseract reads a page the way it finds
    it, so an inverted page yields character soup that looks like text to
    everything downstream. Orientation detection needs a certain amount of
    script on the page and exits non-zero when it finds too little, which is
    the blank and near-blank case and leaves the page as it is.
    """
    completed = subprocess.run(
        [tesseract, str(image), "-", "--psm", "0", "osd"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return 0
    match = _ROTATION.search(completed.stdout)
    if match is None:
        return 0
    return int(match.group(1)) % 360


def degrade_scan(image: PILImage, seed: int) -> PILImage:
    """Make a crisp render look like it came off a feeder.

    A PDF that carries its own text layer renders to clean glyphs, and OCR
    reads it almost perfectly. Paper through a scanner arrives skewed by a
    fraction of a degree, softened by the optics, speckled by the sensor and
    squeezed by JPEG, which is why a model trained on renders meets a
    different distribution the day it sees a real stack. The transforms are
    deliberately mild: this is the same page, scanned, not a harder page.
    """
    from PIL import Image, ImageFilter

    rng = random.Random(seed)
    angle = rng.uniform(-1.2, 1.2)
    turned = image.convert("L").rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)
    blurred = turned.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))
    pixels = blurred.load()
    width, height = blurred.size
    for _ in range(int(width * height * rng.uniform(0.001, 0.004))):
        x, y = rng.randrange(width), rng.randrange(height)
        pixels[x, y] = 0 if rng.random() < 0.5 else 255
    buffer = io.BytesIO()
    blurred.save(buffer, format="JPEG", quality=rng.randint(45, 75))
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("L").copy()


def _word_score(text: str) -> int:
    """Count characters sitting in word-like runs, as a readability proxy.

    Orientation detection needs a certain amount of script and gives up on a
    sparse page, which is exactly the duplex backside carrying two lines. Text
    read upside down still returns characters, but they scatter into short
    fragments, so the longer the runs of letters, the more likely the page is
    the right way up. This needs no dictionary and so no language.
    """
    return sum(len(token) for token in re.findall(r"[^\W\d_]{3,}", text))


def pdf_input(
    input_pdf: Path,
    *,
    languages: str = "eng",
    dpi: int = 200,
    pdftoppm_path: str | None = None,
    tesseract_path: str | None = None,
    degrade_seed: int | None = None,
) -> PdfInput:
    """Render and OCR a PDF without retaining intermediate images on disk.

    `degrade_seed` runs each rendered page through `degrade_scan` first, which
    turns a clean render into something shaped like a scan. It exists to build
    training pages that match what a feeder delivers.
    """
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
            if degrade_seed is not None:
                with Image.open(image_path) as rendered:
                    degrade_scan(rendered, degrade_seed + index).save(image_path)
            rotation = detect_rotation(image_path, tesseract)
            if rotation:
                # Layout boxes and the visual summary have to describe the same
                # upright page the text came from, so the rotation happens once
                # here and every feature is read off the turned image.
                with Image.open(image_path) as image:
                    image.rotate(-rotation, expand=True).save(image_path)
            with Image.open(image_path) as image:
                text, layout = parse_tesseract_tsv(
                    _ocr_tsv(image_path, tesseract, languages), *image.size
                )
            if rotation == 0 and 0 < _word_score(text) < _SPARSE_PAGE_CHARS:
                # A page too sparse for orientation detection can still be
                # upside down, which a feeder produces on every backside of a
                # double-sided pass. Reading it both ways costs one more pass
                # on the few pages that carry almost no text.
                with Image.open(image_path) as image:
                    flipped_path = image_path.with_name(f"flipped-{image_path.name}")
                    image.rotate(180, expand=True).save(flipped_path)
                with Image.open(flipped_path) as flipped:
                    flipped_text, flipped_layout = parse_tesseract_tsv(
                        _ocr_tsv(flipped_path, tesseract, languages), *flipped.size
                    )
                if _word_score(flipped_text) > _word_score(text):
                    image_path.unlink()
                    flipped_path.rename(image_path)
                    text, layout = flipped_text, flipped_layout
                else:
                    flipped_path.unlink()
            page = PageRef(source=source, page=index)
            pages.append(page)
            texts[page] = text
            layouts[page] = layout
            visuals[page] = extract_visual_from_img(image_path.read_bytes())

    stream = Stream(pages=tuple(pages))
    corpus = HfPssCorpus(streams=[stream], _texts=texts, _layouts=layouts, _visuals=visuals)
    return PdfInput(corpus=corpus, stream=stream)
