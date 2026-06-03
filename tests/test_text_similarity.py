from __future__ import annotations

from papercut.eval.runner import evaluate
from papercut.models.baselines.text_similarity import TextSimilarityBaseline
from papercut.streams.resolver import DictResolver
from papercut.streams.types import PageRef, Stream


def test_low_similarity_flagged_as_boundary() -> None:
    pages = (PageRef("d1", 0), PageRef("d1", 1), PageRef("d2", 0))
    resolver = DictResolver(
        {
            pages[0]: "Quarterly results for Acme Corp. Revenue up 12 percent.",
            pages[1]: "Quarterly results for Acme Corp. Outlook remains strong.",
            pages[2]: "Cooking recipes: how to roast a duck with cherry sauce.",
        }
    )
    model = TextSimilarityBaseline(resolver=resolver, threshold=0.15, n=4)
    pred = model.predict_boundaries(Stream(pages=pages))
    assert pred[0] is True
    assert pred[1] is False
    assert pred[2] is True


def test_identical_pages_no_extra_boundary() -> None:
    pages = (PageRef("d", 0), PageRef("d", 1), PageRef("d", 2))
    resolver = DictResolver(dict.fromkeys(pages, "Acme Corp quarterly report fiscal year"))
    model = TextSimilarityBaseline(resolver=resolver, threshold=0.5, n=4)
    pred = model.predict_boundaries(Stream(pages=pages))
    assert pred == (True, False, False)


def test_runs_through_evaluate() -> None:
    pages_a = (PageRef("a", 0), PageRef("a", 1), PageRef("b", 0))
    pages_b = (PageRef("c", 0),)
    resolver = DictResolver(
        {
            pages_a[0]: "doc one page one quarterly",
            pages_a[1]: "doc one page two quarterly",
            pages_a[2]: "doc two completely different content cooking",
            pages_b[0]: "standalone single page document",
        }
    )
    streams = [
        Stream(pages=pages_a, boundaries=(True, False, True)),
        Stream(pages=pages_b, boundaries=(True,)),
    ]
    report = evaluate(TextSimilarityBaseline(resolver=resolver), streams)
    assert report.n_streams == 2
    assert report.stp == 1.0
