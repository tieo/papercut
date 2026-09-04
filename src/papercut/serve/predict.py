"""Run a saved PSS model over scanned PDFs and write clean split documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from papercut.models.baselines.tfidf_xgb_layout_sem_vis import TfIdfXgbLayoutSemVis
from papercut.models.smoothing.blank_gate import BlankPageGated, blank_page_flags
from papercut.serve.pdf_input import pdf_input
from papercut.serve.split_pdf import SplitOutput, write_split_pdfs


@dataclass(frozen=True)
class SplitReport:
    """The prediction and generated documents for one scanned input PDF."""

    boundaries: tuple[bool, ...]
    blank_pages: tuple[bool, ...]
    outputs: tuple[SplitOutput, ...]


class PdfSplitter:
    """Reuse the MiniLM encoder while splitting a sequence of PDF stacks."""

    def __init__(
        self,
        model_path: Path,
        *,
        languages: str = "eng",
        dpi: int = 200,
        threshold: float | None = None,
        pdftoppm_path: str | None = None,
        tesseract_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.languages = languages
        self.dpi = dpi
        self.threshold = threshold
        self.pdftoppm_path = pdftoppm_path
        self.tesseract_path = tesseract_path
        self._encoder = None

    def split(
        self, input_pdf: Path, output_dir: Path, filename_prefix: str | None = None
    ) -> SplitReport:
        """Split one PDF and omit pages classified as blank from its outputs."""
        parsed = pdf_input(
            input_pdf,
            languages=self.languages,
            dpi=self.dpi,
            pdftoppm_path=self.pdftoppm_path,
            tesseract_path=self.tesseract_path,
        )
        model = TfIdfXgbLayoutSemVis.load_with_corpus(
            self.model_path, parsed.corpus, encoder=self._encoder
        )
        if self.threshold is not None:
            model.threshold = self.threshold
        gated = BlankPageGated(submodel=model, corpus=parsed.corpus)
        probs = gated.predict_probs(parsed.stream)
        boundaries = (True, *(p > model.threshold for p in probs[1:]))
        blanks = blank_page_flags(parsed.corpus, parsed.stream)
        self._encoder = model._encoder
        outputs = write_split_pdfs(
            input_pdf,
            output_dir,
            boundaries,
            blanks,
            filename_prefix=filename_prefix or input_pdf.stem,
        )
        return SplitReport(boundaries=boundaries, blank_pages=blanks, outputs=outputs)
