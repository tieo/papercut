from __future__ import annotations

import pytest

from papercut.models.base import ProbabilisticModel
from papercut.models.baselines.text_similarity import TextSimilarityBaseline
from papercut.models.baselines.trivial import EveryPageNewDoc, NeverSplit
from papercut.models.ensembles.late import LateEnsemble
from papercut.streams.resolver import DictResolver
from papercut.streams.types import PageRef, Stream


def _stream() -> tuple[Stream, DictResolver]:
    pages = (PageRef("a", 0), PageRef("a", 1), PageRef("b", 0))
    texts = {
        pages[0]: "Acme Corp quarterly results header letterhead",
        pages[1]: "Acme Corp continued body text similar header",
        pages[2]: "Cooking recipes today roast duck cherry sauce",
    }
    return Stream(pages=pages), DictResolver(texts)


def test_every_page_predict_probs() -> None:
    stream = Stream(pages=(PageRef("x", 0), PageRef("x", 1), PageRef("x", 2)))
    assert EveryPageNewDoc().predict_probs(stream) == (1.0, 1.0, 1.0)


def test_never_split_predict_probs() -> None:
    stream = Stream(pages=(PageRef("x", 0), PageRef("x", 1), PageRef("x", 2)))
    assert NeverSplit().predict_probs(stream) == (1.0, 0.0, 0.0)


def test_text_similarity_predict_probs() -> None:
    stream, resolver = _stream()
    model = TextSimilarityBaseline(resolver=resolver, threshold=0.15, n=4)
    probs = model.predict_probs(stream)
    assert probs[0] == 1.0
    assert probs[1] < probs[2]


def test_text_similarity_boundaries_align_with_probs() -> None:
    stream, resolver = _stream()
    model = TextSimilarityBaseline(resolver=resolver, threshold=0.15, n=4)
    bounds = model.predict_boundaries(stream)
    probs = model.predict_probs(stream)
    assert bounds[0] is True
    for i in range(1, len(stream)):
        assert bounds[i] == (probs[i] > 1 - model.threshold)


def test_every_page_satisfies_probabilistic_protocol() -> None:
    assert isinstance(EveryPageNewDoc(), ProbabilisticModel)


def test_late_ensemble_averages_probs() -> None:
    stream, _resolver = _stream()
    ensemble = LateEnsemble(
        submodels=[EveryPageNewDoc(), NeverSplit()],
    )
    probs = ensemble.predict_probs(stream)
    assert probs[0] == 1.0
    assert probs[1] == 0.5
    assert probs[2] == 0.5


def test_late_ensemble_threshold_decides_boundaries() -> None:
    stream, _resolver = _stream()
    ensemble = LateEnsemble(
        submodels=[EveryPageNewDoc(), NeverSplit()],
        threshold=0.4,
    )
    bounds = ensemble.predict_boundaries(stream)
    assert bounds == (True, True, True)

    ensemble_strict = LateEnsemble(
        submodels=[EveryPageNewDoc(), NeverSplit()],
        threshold=0.6,
    )
    bounds = ensemble_strict.predict_boundaries(stream)
    assert bounds == (True, False, False)


def test_late_ensemble_weights() -> None:
    stream, _resolver = _stream()
    ensemble = LateEnsemble(
        submodels=[EveryPageNewDoc(), NeverSplit()],
        weights=[3.0, 1.0],
    )
    probs = ensemble.predict_probs(stream)
    assert probs[1] == pytest.approx(0.75)
    assert probs[2] == pytest.approx(0.75)


def test_late_ensemble_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LateEnsemble(submodels=[])


def test_late_ensemble_fits_trainable_submodels(tmp_path: object) -> None:
    pytest.importorskip("xgboost")
    from papercut.models.baselines.tfidf_xgb import TfIdfXgb

    pages_a = (PageRef("a", 0), PageRef("a", 1), PageRef("b", 0))
    pages_b = (PageRef("c", 0), PageRef("c", 1), PageRef("d", 0), PageRef("d", 1))
    resolver = DictResolver(
        {
            pages_a[0]: "doc one header letterhead",
            pages_a[1]: "doc one body continues",
            pages_a[2]: "doc two header different",
            pages_b[0]: "case three letterhead alpha",
            pages_b[1]: "case three body continues",
            pages_b[2]: "case four letterhead beta",
            pages_b[3]: "case four body continues",
        }
    )
    train = [
        Stream(pages=pages_a, boundaries=(True, False, True)),
        Stream(pages=pages_b, boundaries=(True, False, True, False)),
    ]
    ensemble = LateEnsemble(
        submodels=[
            TextSimilarityBaseline(resolver=resolver),
            TfIdfXgb(resolver=resolver, n_estimators=10, max_depth=3),
        ],
    )
    ensemble.fit(train)
    bounds = ensemble.predict_boundaries(train[0])
    assert bounds[0] is True
    assert len(bounds) == len(train[0])
