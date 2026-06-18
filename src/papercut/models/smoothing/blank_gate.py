from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from papercut.streams.types import Stream

if TYPE_CHECKING:
    from collections.abc import Sequence

    from papercut.data.loaders.hf import HfPssCorpus
    from papercut.models.base import ProbabilisticModel


def _has_visual_content(
    visual: Sequence[float] | None,
    max_mean_intensity: float,
    min_edge_density: float,
) -> bool:
    """True iff the page image looks like it has any ink on it.

    Uses the visual feature vector from extract_visual_from_img:
      index 1 = mean_intensity in [0, 1] (1.0 = pure white)
      index 3 = edge_density (mean absolute gradient)

    A page with text, an image, a signature, a stamp, or a diagram will
    fail at least one of the thresholds. A truly empty piece of paper
    passes both (very high mean intensity, near-zero edges) and is
    therefore deemed to have no visual content.
    """
    if visual is None or len(visual) < 4:
        return False
    mean_i = float(visual[1])
    edge = float(visual[3])
    return mean_i < max_mean_intensity or edge >= min_edge_density


def _is_blank(
    text: str,
    visual: Sequence[float] | None,
    max_mean_intensity: float,
    min_edge_density: float,
) -> bool:
    """A page is 'blank' iff it has neither OCR text nor any visual content.

    A page with any text at all is not blank. A page with no text but a
    picture, signature, or stamp is also not blank (the visual check picks
    that up). Only a sheet that comes off the scanner essentially white
    on both signals is treated as blank.
    """
    if text and text.strip():
        return False
    return not _has_visual_content(visual, max_mean_intensity, min_edge_density)


@dataclass
class BlankPageGated:
    """Wrap a probabilistic model so blank `curr` pages can never start a new doc.

    Real scans of one-sided originals on duplex hardware produce blank
    backside pages. The base text+layout model treats those blanks as
    'unfamiliar' and over-predicts boundary=True on them. A blank page
    is essentially never the first page of a real document, so we
    override the boundary at index i to False whenever page i is blank.
    Probabilities are also clamped to a small value so downstream
    smoothers don't see a false high-confidence signal.

    The rule only ever turns a predicted True into False; content-to-
    content transitions and probabilities are untouched.
    """

    submodel: ProbabilisticModel
    corpus: HfPssCorpus
    max_mean_intensity: float = 0.99
    min_edge_density: float = 0.003
    blank_prob: float = 0.02
    name: str = field(default="blank_gated")

    def _blanks(self, stream: Stream) -> list[bool]:
        flags: list[bool] = []
        for page in stream.pages:
            text = self.corpus.text(page) or ""
            try:
                visual = self.corpus.visual(page)
            except Exception:
                visual = None
            flags.append(
                _is_blank(text, visual, self.max_mean_intensity, self.min_edge_density)
            )
        return flags

    def fit(self, streams: Sequence[Stream]) -> None:
        if callable(getattr(self.submodel, "fit", None)):
            self.submodel.fit(streams)  # type: ignore[attr-defined]

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        probs = list(self.submodel.predict_probs(stream))
        flags = self._blanks(stream)
        for i in range(1, len(probs)):
            if flags[i]:
                probs[i] = min(probs[i], self.blank_prob)
        return tuple(probs)

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        probs = self.predict_probs(stream)
        threshold = float(getattr(self.submodel, "threshold", 0.5))
        return (True, *(p > threshold for p in probs[1:]))
