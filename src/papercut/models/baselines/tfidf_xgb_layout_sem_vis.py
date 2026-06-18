from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

from papercut.models.baselines.tfidf_xgb_layout_sem import TfIdfXgbLayoutSem
from papercut.streams.types import Stream

if TYPE_CHECKING:
    from pathlib import Path

    from papercut.data.loaders.hf import HfPssCorpus
    from papercut.models.baselines.tfidf_xgb_layout_sem import PageEncoder


class TfIdfXgbLayoutSemVis(TfIdfXgbLayoutSem):
    """Sem model plus cross-page visual similarity features.

    Adds 69 dense features per page-pair: prev_vis (17), curr_vis (17),
    diff (17), abs_diff (17), and visual cosine similarity (1). The visual
    vector comes from corpus.visual() — aspect ratio, grayscale intensity
    statistics, edge density, and a 3x3 spatial intensity grid.

    _current_visuals is set before every _build_features call. fit() is
    overridden to set it in the feature-build loop (parent fit uses two
    separate loops; the stash would otherwise point to the wrong stream).
    predict_probs/_gather path is fine: _gather sets the stash then
    _build_features consumes it in the same call.
    """

    name = "tfidf_xgb_layout_sem_vis"

    def __init__(
        self,
        corpus: HfPssCorpus,
        encoder: PageEncoder | None = None,
        **layout_kwargs: object,
    ) -> None:
        super().__init__(corpus=corpus, encoder=encoder, **layout_kwargs)
        self._current_visuals: np.ndarray | None = None

    def _gather_visuals(self, stream: Stream) -> np.ndarray:
        return np.asarray(
            [self.corpus.visual(p) for p in stream.pages], dtype=np.float32
        )

    def _gather(self, stream: Stream) -> tuple[list[str], np.ndarray]:
        texts, layouts = super()._gather(stream)
        self._current_visuals = self._gather_visuals(stream)
        return texts, layouts

    def _build_features(self, texts: list[str], layouts: np.ndarray):
        base = super()._build_features(texts, layouts)
        vis = self._current_visuals
        if vis is None or len(vis) < 2:
            return base
        prev_v = vis[:-1]
        curr_v = vis[1:]
        diff = (prev_v - curr_v).astype(np.float32)
        adiff = np.abs(diff)
        vnorm_p = np.linalg.norm(prev_v, axis=1, keepdims=True) + 1e-9
        vnorm_c = np.linalg.norm(curr_v, axis=1, keepdims=True) + 1e-9
        vcos = (np.sum(prev_v * curr_v, axis=1, keepdims=True) / (vnorm_p * vnorm_c)).astype(
            np.float32
        )
        vis_block = np.hstack([prev_v, curr_v, diff, adiff, vcos])
        return hstack([base, csr_matrix(vis_block)]).tocsr()

    def fit(self, streams: Sequence[Stream]) -> None:
        streams = list(streams)
        all_truncated: list[str] = []
        per_stream: list[tuple[list[str], np.ndarray, np.ndarray]] = []
        for stream in streams:
            if stream.boundaries is None:
                raise ValueError("Cannot fit on unlabeled stream")
            texts, layouts = super(TfIdfXgbLayoutSemVis, self)._gather(stream)
            vis = self._gather_visuals(stream)
            per_stream.append((texts, layouts, vis))
            all_truncated.extend(self._truncate(t) for t in texts)
        if not all_truncated:
            raise ValueError("No texts available")
        self.vectorizer.fit(all_truncated)

        blocks = []
        labels: list[int] = []
        for stream, (texts, layouts, vis) in zip(streams, per_stream, strict=True):
            assert stream.boundaries is not None
            if len(texts) < 2:
                continue
            self._current_visuals = vis
            blocks.append(self._build_features(texts, layouts))
            labels.extend(1 if b else 0 for b in stream.boundaries[1:])
        if not blocks:
            raise ValueError("Need at least one multi-page stream to fit")
        x_train = vstack(blocks).tocsr()
        y_train = np.asarray(labels, dtype=np.int32)
        self.model.fit(x_train, y_train)
        self._fitted = True

    @classmethod
    def load_with_corpus(
        cls,
        path: str | Path,
        corpus: HfPssCorpus,
        encoder: PageEncoder | None = None,
    ) -> TfIdfXgbLayoutSemVis:
        instance = super().load_with_corpus(path, corpus, encoder=encoder)
        instance._current_visuals = None
        return instance  # type: ignore[return-value]
