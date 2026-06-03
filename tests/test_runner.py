from __future__ import annotations

import pytest

from papercut.eval.runner import evaluate
from papercut.models.baselines.trivial import EveryPageNewDoc, NeverSplit
from papercut.streams.types import PageRef, Stream


def _stream(boundaries: tuple[bool, ...]) -> Stream:
    pages = tuple(PageRef(source="fixture", page=i) for i in range(len(boundaries)))
    return Stream(pages=pages, boundaries=boundaries)


def _streams() -> list[Stream]:
    return [
        _stream((True, False, False, True, False)),
        _stream((True, False, True, False, False)),
        _stream((True,)),
    ]


def test_every_page_new_doc_perfect_recall() -> None:
    report = evaluate(EveryPageNewDoc(), _streams())
    assert report.n_streams == 3
    assert 0.0 < report.page_f1_mean <= 1.0
    assert report.stp < 1.0


def test_never_split_zero_recall_on_multidoc() -> None:
    report = evaluate(NeverSplit(), _streams())
    assert report.stp == pytest.approx(1 / 3)
    assert report.page_f1_mean < 0.5


def test_evaluate_rejects_unlabeled() -> None:
    pages = (PageRef(source="fixture", page=0), PageRef(source="fixture", page=1))
    with pytest.raises(ValueError, match="unlabeled"):
        evaluate(EveryPageNewDoc(), [Stream(pages=pages)])


def test_evaluate_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate(EveryPageNewDoc(), [])


def test_evaluate_detects_mismatched_prediction_length() -> None:
    class Buggy:
        name = "buggy"

        def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
            return (True,)

    with pytest.raises(ValueError, match="returned"):
        evaluate(Buggy(), [_stream((True, False, True))])
