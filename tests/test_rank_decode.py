from __future__ import annotations

from dataclasses import dataclass

from papercut.models.smoothing.rank_decode import ExpectedCount, RankNormalized, rank_quantiles
from papercut.streams.types import PageRef, Stream


@dataclass
class FixedProbs:
    probs: dict[str, tuple[float, ...]]
    name: str = "fixed"

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        return self.probs[stream.pages[0].source]

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        return (True, *(p > 0.5 for p in self.predict_probs(stream)[1:]))


def _stream(source: str, pages: int) -> Stream:
    return Stream(
        pages=tuple(PageRef(source=source, page=i) for i in range(pages)),
        boundaries=(True, *([False] * (pages - 1))),
    )


def test_rank_quantiles_span_zero_to_one_in_order() -> None:
    quantiles = rank_quantiles([0.2, 0.9, 0.5])
    assert quantiles[0] == 0.0
    assert quantiles[1] == 1.0
    assert 0.0 < quantiles[2] < 1.0


def test_rank_quantiles_average_ties() -> None:
    assert rank_quantiles([0.4, 0.4]) == (0.5, 0.5)


def test_saturated_scores_still_separate_after_ranking() -> None:
    stream = _stream("s", 5)
    model = RankNormalized(submodel=FixedProbs({"s": (1.0, 0.99, 0.999, 0.98, 0.9999)}), threshold=0.75)
    predicted = model.predict_boundaries(stream)
    assert predicted[0] is True
    assert predicted[4] is True  # highest of the saturated scores
    assert predicted[3] is False  # lowest stays closed


def test_expected_count_opens_as_many_documents_as_length_suggests() -> None:
    stream = _stream("s", 12)
    model = ExpectedCount(
        submodel=FixedProbs({"s": (1.0, 0.1, 0.9, 0.2, 0.8, 0.15, 0.3, 0.05, 0.25, 0.4, 0.35, 0.02)}),
        pages_per_document=4.0,
    )
    predicted = model.predict_boundaries(stream)
    assert sum(predicted) == 3  # twelve pages, four per document
    assert predicted[2] and predicted[4]  # the two strongest scores


def test_expected_count_learns_its_prior_from_labels() -> None:
    labeled = Stream(
        pages=tuple(PageRef(source="t", page=i) for i in range(6)),
        boundaries=(True, False, True, False, False, False),
    )
    model = ExpectedCount(submodel=FixedProbs({"t": (1.0,) * 6}))
    model.fit_length([labeled])
    assert model.pages_per_document == 3.0


def test_expected_count_always_opens_the_first_page() -> None:
    stream = _stream("s", 3)
    model = ExpectedCount(submodel=FixedProbs({"s": (0.0, 0.0, 0.0)}), pages_per_document=99.0)
    assert model.predict_boundaries(stream)[0] is True
