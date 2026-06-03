from __future__ import annotations

import pytest

from papercut.eval.metrics import (
    boundaries_to_doc_ids,
    boundaries_to_spans,
    mndd,
    page_metrics,
    panoptic_quality,
    stp,
)


def test_boundaries_to_spans() -> None:
    assert boundaries_to_spans([True, False, False, True, False]) == [(0, 3), (3, 5)]
    assert boundaries_to_spans([True]) == [(0, 1)]
    assert boundaries_to_spans([True, True, True]) == [(0, 1), (1, 2), (2, 3)]


def test_boundaries_to_doc_ids() -> None:
    assert boundaries_to_doc_ids([True, False, False, True, False]) == [0, 0, 0, 1, 1]
    assert boundaries_to_doc_ids([True, True, True]) == [0, 1, 2]


def test_page_metrics_perfect() -> None:
    bnd = [True, False, False, True, False, True]
    prf = page_metrics(bnd, bnd)
    assert prf.precision == 1.0
    assert prf.recall == 1.0
    assert prf.f1 == 1.0


def test_page_metrics_partial() -> None:
    true = [True, False, False, True, False, True]
    pred = [True, False, True, False, False, True]
    prf = page_metrics(true, pred)
    assert prf.precision == pytest.approx(1 / 2)
    assert prf.recall == pytest.approx(1 / 2)
    assert prf.f1 == pytest.approx(0.5)


def test_page_metrics_first_page_excluded() -> None:
    """If only the trivial first-page boundary is correct, F1 is 0."""
    true = [True, False, True]
    pred = [True, False, False]
    prf = page_metrics(true, pred)
    assert prf.recall == 0.0
    assert prf.f1 == 0.0


def test_panoptic_quality_perfect() -> None:
    bnd = [True, False, False, True, False, True]
    pq = panoptic_quality(bnd, bnd)
    assert pq.rq == 1.0
    assert pq.sq == 1.0
    assert pq.pq == 1.0


def test_panoptic_quality_off_by_one() -> None:
    """A 5-page doc split as 4+1 should still match the 5-page true (IoU 4/5)."""
    true = [True, False, False, False, False]
    pred = [True, False, False, False, True]
    pq = panoptic_quality(true, pred)
    assert pq.rq == pytest.approx(2 / 3)
    assert pq.sq == pytest.approx(0.8)


def test_panoptic_quality_no_match() -> None:
    """A 2-page doc split into singletons has IoU 0.5 with neither match."""
    true = [True, False]
    pred = [True, True]
    pq = panoptic_quality(true, pred)
    assert pq.rq == 0.0
    assert pq.sq == 0.0
    assert pq.pq == 0.0


def test_stp_all_exact() -> None:
    pairs = [([True, False], [True, False]), ([True, True], [True, True])]
    assert stp(pairs) == 1.0


def test_stp_half() -> None:
    pairs = [([True, False], [True, False]), ([True, True], [True, False])]
    assert stp(pairs) == 0.5


def test_mndd_perfect() -> None:
    bnd = [True, False, False, True, False]
    assert mndd(bnd, bnd) == 0


def test_mndd_one_misplaced() -> None:
    """One page from doc 0 dragged into doc 1: that page is wrong, plus the
    page that should be in doc 0 but is now in doc 1 is also wrong if it
    crossed over. For a simple split error, count is one page misplaced."""
    true = [True, False, False, True, False]
    pred = [True, False, True, False, False]
    assert mndd(true, pred) == 1


def test_mndd_complete_disagreement() -> None:
    true = [True, False, False, False]
    pred = [True, True, True, True]
    assert mndd(true, pred) == 3


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        page_metrics([True, False], [True])


def test_stp_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        stp([])
