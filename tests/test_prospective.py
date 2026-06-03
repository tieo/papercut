from __future__ import annotations

import pytest

from papercut.eval.prospective import Slice, format_results, walk_forward
from papercut.models.baselines.trivial import EveryPageNewDoc, NeverSplit
from papercut.streams.types import PageRef, Stream


def _stream(boundaries: tuple[bool, ...]) -> Stream:
    pages = tuple(PageRef(source="fx", page=i) for i in range(len(boundaries)))
    return Stream(pages=pages, boundaries=boundaries)


def _slices() -> list[Slice]:
    return [
        Slice("2024-01", [_stream((True, False, False, True, False))]),
        Slice("2024-02", [_stream((True, False, True, False, False))]),
        Slice("2024-03", [_stream((True, True, True))]),
    ]


def test_walk_forward_produces_expected_shape() -> None:
    models = [EveryPageNewDoc(), NeverSplit()]
    results = walk_forward(models, _slices())
    assert len(results) == 4
    assert {r.test_slice for r in results} == {"2024-02", "2024-03"}
    for r in results:
        assert r.test_slice not in r.train_slices


def test_walk_forward_train_slices_grow() -> None:
    results = walk_forward([NeverSplit()], _slices())
    by_test = {r.test_slice: r for r in results}
    assert by_test["2024-02"].train_slices == ("2024-01",)
    assert by_test["2024-03"].train_slices == ("2024-01", "2024-02")


def test_walk_forward_requires_two_slices() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        walk_forward([NeverSplit()], _slices()[:1])


def test_format_results_renders_table() -> None:
    results = walk_forward([NeverSplit()], _slices())
    rendered = format_results(results)
    assert "never_split" in rendered
    assert "2024-02" in rendered
    assert "page_f1" in rendered
