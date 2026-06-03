from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from papercut.models.baselines.multilingual_minilm import MultilingualMiniLM
from papercut.streams.resolver import DictResolver
from papercut.streams.types import PageRef, Stream


class _FakeEncoder:
    """Deterministic fake encoder for tests. Encodes by hashing tokens."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def __call__(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, hash(tok) % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


def _build_corpus(
    n_streams: int = 12,
) -> tuple[list[Stream], DictResolver]:
    streams: list[Stream] = []
    texts: dict[PageRef, str] = {}
    for s in range(n_streams):
        pages = tuple(PageRef(source=f"s{s}", page=i) for i in range(5))
        page_texts = [
            f"alpha letterhead unique words s{s} a a a a a",
            f"alpha continued body s{s} a a a a a",
            f"beta header completely different words s{s} b b b b b",
            f"beta body continues s{s} b b b b b",
            f"gamma standalone topic s{s} c c c c c",
        ]
        for p, t in zip(pages, page_texts, strict=True):
            texts[p] = t
        streams.append(Stream(pages=pages, boundaries=(True, False, True, False, True)))
    return streams, DictResolver(texts)


def test_fits_with_fake_encoder() -> None:
    streams, resolver = _build_corpus()
    model = MultilingualMiniLM(resolver=resolver, encoder=_FakeEncoder())
    model.fit(streams[:8])
    pred = model.predict_boundaries(streams[8])
    assert len(pred) == len(streams[8])
    assert pred[0] is True


def test_predict_probs_aligned_to_pages() -> None:
    streams, resolver = _build_corpus()
    model = MultilingualMiniLM(resolver=resolver, encoder=_FakeEncoder())
    model.fit(streams[:8])
    probs = model.predict_probs(streams[8])
    assert len(probs) == len(streams[8])
    assert probs[0] == 1.0
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_fit_required_before_predict() -> None:
    _, resolver = _build_corpus()
    model = MultilingualMiniLM(resolver=resolver, encoder=_FakeEncoder())
    pages = (PageRef("x", 0), PageRef("x", 1))
    with pytest.raises(RuntimeError, match="must be fit"):
        model.predict_probs(Stream(pages=pages))


def test_save_load_round_trip(tmp_path: Path) -> None:
    streams, resolver = _build_corpus()
    model = MultilingualMiniLM(resolver=resolver, encoder=_FakeEncoder())
    model.fit(streams[:8])
    target = tmp_path / "minilm.pkl"
    model.save(target)

    restored = MultilingualMiniLM.load_with_resolver(
        target, resolver=resolver, encoder=_FakeEncoder()
    )
    pred_a = model.predict_boundaries(streams[8])
    pred_b = restored.predict_boundaries(streams[8])
    assert pred_a == pred_b
