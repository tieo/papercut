from __future__ import annotations

from pathlib import Path

import pytest

from papercut.eval.runner import evaluate
from papercut.models.baselines.tfidf_xgb import TfIdfXgb
from papercut.streams.resolver import DictResolver
from papercut.streams.types import PageRef, Stream


def _stream(
    prefix: str, doc_boundaries: list[int], length: int
) -> tuple[Stream, dict[PageRef, str]]:
    """Create a stream of `length` pages with True boundaries at given indices.

    Page text is `prefix-<doc_idx>-<page_idx>` so pages of the same doc share
    a strong common substring (the doc index) and pages of different docs
    differ on that substring. Makes the task learnable for a tiny test.
    """
    pages = tuple(PageRef(source=f"{prefix}", page=i) for i in range(length))
    boundaries = tuple(i in set(doc_boundaries) for i in range(length))
    texts: dict[PageRef, str] = {}
    doc_idx = -1
    for i, p in enumerate(pages):
        if boundaries[i]:
            doc_idx += 1
        common = f"docid{doc_idx}-unique-letterhead-substring-{doc_idx * 47 % 31}"
        texts[p] = f"{common} page-{i} body content varies a little here {i * 13 % 7}"
    return Stream(pages=pages, boundaries=boundaries), texts


def _build_corpus(n_streams: int = 20) -> tuple[list[Stream], DictResolver]:
    streams: list[Stream] = []
    texts: dict[PageRef, str] = {}
    for s in range(n_streams):
        boundaries = [0]
        cursor = 0
        for _ in range(3):
            length = 2 + (s % 3)
            cursor += length
            if cursor < 12:
                boundaries.append(cursor)
        total = max(boundaries) + 2 + (s % 4)
        stream, page_texts = _stream(prefix=f"stream{s}", doc_boundaries=boundaries, length=total)
        streams.append(stream)
        texts.update(page_texts)
    return streams, DictResolver(texts)


def test_tfidf_xgb_fits_and_predicts_shape() -> None:
    streams, resolver = _build_corpus(n_streams=15)
    model = TfIdfXgb(resolver=resolver, n_estimators=20, max_depth=3)
    model.fit(streams)
    pred = model.predict_boundaries(streams[0])
    assert len(pred) == len(streams[0])
    assert pred[0] is True


def test_tfidf_xgb_beats_trivial_on_learnable_corpus() -> None:
    streams, resolver = _build_corpus(n_streams=40)
    train, test = streams[:30], streams[30:]
    model = TfIdfXgb(resolver=resolver, n_estimators=50, max_depth=4)
    model.fit(train)
    report = evaluate(model, test)
    assert report.page_f1_mean > 0.6
    assert report.stp > 0.0


def test_tfidf_xgb_predict_before_fit_raises() -> None:
    _, resolver = _build_corpus(n_streams=2)
    model = TfIdfXgb(resolver=resolver)
    pages = (PageRef("stream0", 0), PageRef("stream0", 1))
    with pytest.raises(RuntimeError, match="must be fit"):
        model.predict_boundaries(Stream(pages=pages))


def test_tfidf_xgb_save_and_load_roundtrip(tmp_path: Path) -> None:
    streams, resolver = _build_corpus(n_streams=10)
    model = TfIdfXgb(resolver=resolver, n_estimators=10, max_depth=3)
    model.fit(streams)
    pred_before = model.predict_boundaries(streams[0])

    model_path = tmp_path / "tfidf_xgb.pkl"
    model.save(str(model_path))
    restored = TfIdfXgb.load_with_resolver(str(model_path), resolver)
    pred_after = restored.predict_boundaries(streams[0])
    assert pred_after == pred_before


def test_tfidf_xgb_handles_singleton_stream() -> None:
    streams, resolver = _build_corpus(n_streams=5)
    model = TfIdfXgb(resolver=resolver, n_estimators=10, max_depth=3)
    model.fit(streams)
    pred = model.predict_boundaries(Stream(pages=(PageRef("stream0", 0),)))
    assert pred == (True,)
