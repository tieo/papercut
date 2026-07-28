"""Draw representative Papercut CLI reports for Viewbook."""

from __future__ import annotations

from pathlib import Path
from subprocess import check_output

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path(__file__).with_name("img")


def font(pattern: str, size: int) -> ImageFont.FreeTypeFont:
    path = check_output(["fc-match", "-f", "%{file}", pattern], text=True).strip()
    return ImageFont.truetype(path, size)


FONT = font("DejaVu Sans Mono", 22)
TITLE = font("DejaVu Sans", 32)


REPORTS = {
    "split-report": {
        "normal": ["$ papercut serve split-pdf scan.pdf split/", "Split 12 pages into 4 documents", "Removed 2 confirmed blank pages", "", "scan_001.pdf   source pages 1, 2", "scan_002.pdf   source pages 3, 4, 5", "scan_003.pdf   source pages 6", "scan_004.pdf   source pages 7, 8, 9, 10"],
        "loading": ["$ papercut serve split-pdf scan.pdf split/", "", "Rendering pages", "OCRing text and layout", "Classifying page boundaries", "", "Working..."],
        "empty": ["$ papercut serve split-pdf scan.pdf split/", "", "No output documents were written.", "The input contains no pages with content."],
        "failed": ["$ papercut serve split-pdf scan.pdf split/", "", "Error: Tesseract could not OCR page 4.", "The source PDF remains unchanged.", "Fix the OCR command and run again."],
    },
    "inbox-report": {
        "normal": ["$ papercut serve process-inbox inbox/ consume/", "Processed 3 input PDFs", "", "scan-a1b2.pdf   2 documents, 1 blank page removed", "scan-c3d4.pdf   1 document, 0 blank pages removed", "scan-e5f6.pdf   4 documents, 2 blank pages removed"],
        "loading": ["$ papercut serve process-inbox inbox/ consume/", "", "Checking input versions", "Processing unseen PDFs", "", "Working..."],
        "empty": ["$ papercut serve process-inbox inbox/ consume/", "", "Processed 0 input PDFs.", "Every input is already recorded at its current content hash."],
        "failed": ["$ papercut serve process-inbox inbox/ consume/", "", "Error: one input could not be processed.", "It remains in the inbox for a later retry."],
    },
}


def draw(view: str, state: str, theme: str, shape: str) -> None:
    width, height = (1440, 820) if shape == "wide" else (680, 1040)
    dark = theme == "dark"
    background = "#111827" if dark else "#f8fafc"
    foreground = "#e2e8f0" if dark else "#172033"
    panel = "#1e293b" if dark else "#ffffff"
    accent = "#38bdf8" if dark else "#0369a1"
    image = Image.new("RGB", (width, height), background)
    canvas = ImageDraw.Draw(image)
    margin = 52 if shape == "wide" else 34
    canvas.rounded_rectangle((margin, margin, width - margin, height - margin), radius=18, fill=panel)
    title = "Split report" if view == "split-report" else "Inbox processing report"
    canvas.text((margin + 32, margin + 28), title, font=TITLE, fill=foreground)
    state_label = "Ready" if state == "normal" else state.title()
    canvas.rounded_rectangle((width - margin - 150, margin + 28, width - margin - 32, margin + 70), radius=16, fill=accent)
    canvas.text((width - margin - 132, margin + 37), state_label, font=FONT, fill=panel)
    y = margin + 116
    for line in REPORTS[view][state]:
        fill = accent if line.startswith("$") else foreground
        canvas.text((margin + 32, y), line, font=FONT, fill=fill)
        y += 40
    suffix = f"{view}-{'' if state == 'normal' else state + '-'}{shape}-{theme}.png"
    image.save(OUTPUT / suffix)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for view in REPORTS:
        for state in REPORTS[view]:
            for theme in ("light", "dark"):
                for shape in ("wide", "phone"):
                    draw(view, state, theme, shape)


if __name__ == "__main__":
    main()
