from __future__ import annotations

from pathlib import Path

from reportlab.pdfgen.canvas import Canvas


def make_pdf(path: Path, n_pages: int, text: str = "fixture") -> Path:
    """Write a tiny `n_pages`-long PDF to `path` and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(path))
    for i in range(n_pages):
        canvas.drawString(100, 100, f"{text} page {i + 1}")
        canvas.showPage()
    canvas.save()
    return path
