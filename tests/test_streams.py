from __future__ import annotations

import pytest

from papercut.streams.types import PageRef, Stream


def _pages(n: int) -> tuple[PageRef, ...]:
    return tuple(PageRef(source="fixture", page=i) for i in range(n))


def test_unlabeled_stream() -> None:
    s = Stream(pages=_pages(5))
    assert len(s) == 5
    assert s.boundaries is None


def test_labeled_stream_counts_documents() -> None:
    s = Stream(pages=_pages(5), boundaries=(True, False, False, True, False))
    assert s.n_documents == 2


def test_first_page_must_start_document() -> None:
    with pytest.raises(ValueError, match="boundaries\\[0\\]"):
        Stream(pages=_pages(3), boundaries=(False, True, False))


def test_boundaries_length_must_match_pages() -> None:
    with pytest.raises(ValueError, match="length"):
        Stream(pages=_pages(3), boundaries=(True, False))


def test_empty_stream_rejected() -> None:
    with pytest.raises(ValueError, match="at least one page"):
        Stream(pages=())
