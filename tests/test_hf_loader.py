from __future__ import annotations

import pytest

from papercut.data.loaders.hf import (
    HfPssSchema,
    assert_resolver_protocol,
    corpus_from_rows,
)
from papercut.streams.resolver import PageResolver
from papercut.streams.types import PageRef


def _rows() -> list[dict[str, object]]:
    return [
        {
            "stream_id": "S0",
            "page_ids": ["S0/p0", "S0/p1", "S0/p2", "S0/p3"],
            "boundaries": [True, False, True, False],
            "texts": ["alpha", "alpha continued", "beta", "beta continued"],
        },
        {
            "stream_id": "S1",
            "page_ids": ["S1/p0", "S1/p1"],
            "boundaries": [True, False],
            "texts": ["gamma", "gamma cont"],
        },
    ]


def _schema() -> HfPssSchema:
    return HfPssSchema(
        boundaries_col="boundaries",
        stream_id_col="stream_id",
        page_ids_col="page_ids",
        text_col="texts",
    )


def test_corpus_from_rows_builds_streams() -> None:
    corpus = corpus_from_rows(_rows(), _schema(), "test_ns")
    assert len(corpus.streams) == 2
    assert corpus.streams[0].boundaries == (True, False, True, False)
    assert corpus.streams[0].pages[0] == PageRef(source="test_ns:S0/p0", page=0)
    assert corpus.text(corpus.streams[0].pages[1]) == "alpha continued"


def test_corpus_respects_max_streams() -> None:
    corpus = corpus_from_rows(_rows(), _schema(), "ns", max_streams=1)
    assert len(corpus.streams) == 1


def test_corpus_satisfies_resolver_protocol() -> None:
    corpus = corpus_from_rows(_rows(), _schema(), "ns")
    assert isinstance(corpus, PageResolver)
    assert_resolver_protocol(corpus)


def test_missing_text_for_page_raises() -> None:
    schema = HfPssSchema(boundaries_col="boundaries", stream_id_col="stream_id")
    rows = [{"stream_id": "S0", "boundaries": [True, False]}]
    corpus = corpus_from_rows(rows, schema, "ns")
    with pytest.raises(KeyError, match="No text"):
        corpus.text(corpus.streams[0].pages[0])


def test_synthetic_id_generation_when_no_page_ids() -> None:
    schema = HfPssSchema(boundaries_col="boundaries", stream_id_col="stream_id")
    rows = [{"stream_id": "X9", "boundaries": [True, False, False]}]
    corpus = corpus_from_rows(rows, schema, "ns")
    assert corpus.streams[0].pages[0].source == "ns:X9/0"
    assert corpus.streams[0].pages[2].source == "ns:X9/2"


def test_mismatched_text_length_raises() -> None:
    schema = HfPssSchema(
        boundaries_col="boundaries",
        stream_id_col="stream_id",
        text_col="texts",
    )
    rows = [{"stream_id": "S0", "boundaries": [True, False], "texts": ["only one"]}]
    with pytest.raises(ValueError, match="text length"):
        corpus_from_rows(rows, schema, "ns")


def test_corpus_works_as_resolver_with_tfidf_xgb() -> None:
    """Smoke: a real model can train+predict on an HfPssCorpus."""
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    from papercut.models.baselines.tfidf_xgb import TfIdfXgb

    rich_rows = []
    for s in range(8):
        rich_rows.append(
            {
                "stream_id": f"S{s}",
                "page_ids": [f"S{s}/p{i}" for i in range(5)],
                "boundaries": [True, False, True, False, True],
                "texts": [
                    f"acme corp letterhead {s} chapter 1 introduction overview",
                    f"acme corp body content section continues {s} discussion",
                    f"cooking recipe roast duck cherry sauce {s} ingredient list",
                    f"cooking body ingredient continues simmer reduce {s}",
                    f"weather report cloudy {s} precipitation forecast tomorrow",
                ],
            }
        )
    corpus = corpus_from_rows(rich_rows, _schema(), "ns")
    model = TfIdfXgb(resolver=corpus, n_estimators=20, max_depth=3)
    model.fit(corpus.streams[:6])
    pred = model.predict_boundaries(corpus.streams[6])
    assert len(pred) == len(corpus.streams[6])
    assert pred[0] is True
