from __future__ import annotations

from papercut.models.smoothing.viterbi import SequenceSmoothed
from papercut.streams.types import PageRef, Stream


class _FixedProbsModel:
    """Submodel that returns predetermined probabilities for a stream."""

    name = "fixed"

    def __init__(self, mapping: dict[tuple[str, int], float]) -> None:
        self.mapping = mapping

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        return tuple(self.mapping[(p.source, p.page)] for p in stream.pages)

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        return tuple(self.mapping[(p.source, p.page)] > 0.5 for p in stream.pages)


def _stream(n: int) -> Stream:
    return Stream(pages=tuple(PageRef("x", i) for i in range(n)))


def test_viterbi_preserves_first_page_boundary() -> None:
    probs = {("x", 0): 0.99, ("x", 1): 0.0, ("x", 2): 0.0}
    sm = SequenceSmoothed(submodel=_FixedProbsModel(probs))
    bounds = sm.predict_boundaries(_stream(3))
    assert bounds[0] is True


def test_viterbi_smooths_isolated_spike() -> None:
    """A single high-prob page surrounded by low-prob neighbors with high
    transition cost gets smoothed away."""
    probs = {
        ("x", 0): 1.0,
        ("x", 1): 0.05,
        ("x", 2): 0.55,
        ("x", 3): 0.05,
        ("x", 4): 0.05,
    }
    sm = SequenceSmoothed(
        submodel=_FixedProbsModel(probs),
        boundary_rate=0.05,
        same_to_new=0.05,
        new_to_new=0.05,
    )
    bounds = sm.predict_boundaries(_stream(5))
    assert bounds[2] is False


def test_viterbi_keeps_strong_evidence() -> None:
    probs = {
        ("x", 0): 1.0,
        ("x", 1): 0.05,
        ("x", 2): 0.95,
        ("x", 3): 0.05,
        ("x", 4): 0.05,
    }
    sm = SequenceSmoothed(
        submodel=_FixedProbsModel(probs),
        boundary_rate=0.2,
        same_to_new=0.2,
        new_to_new=0.05,
    )
    bounds = sm.predict_boundaries(_stream(5))
    assert bounds[2] is True


def test_viterbi_fit_calibrates_boundary_rate() -> None:
    sm = SequenceSmoothed(submodel=_FixedProbsModel({}))
    training = [
        Stream(
            pages=tuple(PageRef("y", i) for i in range(10)),
            boundaries=(True, False, False, True, False, False, False, True, False, False),
        ),
    ]
    sm.fit(training)
    assert 0.0 < sm.boundary_rate < 0.5
    assert abs(sm.boundary_rate - 2 / 9) < 0.01


def test_viterbi_single_page_stream() -> None:
    sm = SequenceSmoothed(submodel=_FixedProbsModel({("x", 0): 1.0}))
    bounds = sm.predict_boundaries(_stream(1))
    assert bounds == (True,)
