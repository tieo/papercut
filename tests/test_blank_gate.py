from __future__ import annotations

from dataclasses import dataclass, field

from papercut.models.smoothing.blank_gate import BlankPageGated
from papercut.streams.types import PageRef, Stream

WHITE_PAPER = [1.0, 1.0, 0.0, 0.0] + [0.0] * 13
CONTENT_PAGE = [1.0, 0.7, 0.2, 0.05] + [0.0] * 13


@dataclass
class _Corpus:
    texts: dict[tuple[str, int], str] = field(default_factory=dict)
    visuals: dict[tuple[str, int], list[float]] = field(default_factory=dict)

    def text(self, page: PageRef) -> str:
        return self.texts.get((page.source, page.page), "")

    def visual(self, page: PageRef) -> list[float]:
        return self.visuals.get((page.source, page.page), [0.0] * 17)


class _FixedModel:
    name = "fixed"
    threshold = 0.5

    def __init__(self, probs: list[float]) -> None:
        self._probs = probs

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        return tuple(self._probs)

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        return (True, *(p > 0.5 for p in self._probs[1:]))


def _stream(n: int) -> Stream:
    return Stream(pages=tuple(PageRef("u", i) for i in range(n)))


def test_truly_blank_page_overrides_high_prob_boundary_to_false() -> None:
    corpus = _Corpus(
        texts={("u", 0): "content " * 50, ("u", 1): "", ("u", 2): "content " * 50},
        visuals={
            ("u", 0): CONTENT_PAGE,
            ("u", 1): WHITE_PAPER,
            ("u", 2): CONTENT_PAGE,
        },
    )
    base = _FixedModel([1.0, 0.9, 0.9])
    gated = BlankPageGated(submodel=base, corpus=corpus)
    assert gated.predict_boundaries(_stream(3)) == (True, False, True)


def test_any_text_means_not_blank_even_with_white_visual() -> None:
    """A single OCR word is enough to count as a real page."""
    corpus = _Corpus(
        texts={("u", 0): "doc one " * 30, ("u", 1): "x"},
        visuals={("u", 0): CONTENT_PAGE, ("u", 1): WHITE_PAPER},
    )
    base = _FixedModel([1.0, 0.95])
    gated = BlankPageGated(submodel=base, corpus=corpus)
    assert gated.predict_boundaries(_stream(2)) == (True, True)


def test_image_only_page_is_not_blank() -> None:
    """No OCR text but the visual signal shows ink: still a real page."""
    corpus = _Corpus(
        texts={("u", 0): "doc one " * 30, ("u", 1): ""},
        visuals={("u", 0): CONTENT_PAGE, ("u", 1): CONTENT_PAGE},
    )
    base = _FixedModel([1.0, 0.95])
    gated = BlankPageGated(submodel=base, corpus=corpus)
    assert gated.predict_boundaries(_stream(2)) == (True, True)


def test_whitespace_only_text_with_white_visual_is_blank() -> None:
    corpus = _Corpus(
        texts={("u", 0): "content " * 50, ("u", 1): "   \n\t "},
        visuals={("u", 0): CONTENT_PAGE, ("u", 1): WHITE_PAPER},
    )
    base = _FixedModel([1.0, 0.95])
    gated = BlankPageGated(submodel=base, corpus=corpus)
    assert gated.predict_boundaries(_stream(2)) == (True, False)


def test_blank_gate_leaves_content_pages_alone() -> None:
    corpus = _Corpus(
        texts={("u", 0): "lots " * 30, ("u", 1): "lots " * 30, ("u", 2): "lots " * 30},
        visuals={k: CONTENT_PAGE for k in [("u", 0), ("u", 1), ("u", 2)]},
    )
    base = _FixedModel([1.0, 0.8, 0.2])
    gated = BlankPageGated(submodel=base, corpus=corpus)
    assert gated.predict_boundaries(_stream(3)) == (True, True, False)


def test_blank_gate_clamps_prob_for_blank_pages() -> None:
    corpus = _Corpus(
        texts={("u", 0): "lots " * 30, ("u", 1): ""},
        visuals={("u", 0): CONTENT_PAGE, ("u", 1): WHITE_PAPER},
    )
    base = _FixedModel([1.0, 0.95])
    gated = BlankPageGated(submodel=base, corpus=corpus, blank_prob=0.01)
    probs = gated.predict_probs(_stream(2))
    assert probs[1] <= 0.01
