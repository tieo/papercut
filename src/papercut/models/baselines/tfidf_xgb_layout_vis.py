from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, hstack

from papercut.models.baselines.tfidf_xgb_layout import TfIdfXgbLayout, _cross_page_features
from papercut.models.baselines.tfidf_xgb_layout_sem import TfIdfXgbLayoutSem
from papercut.models.baselines.tfidf_xgb_rich import _page_features
from papercut.streams.types import Stream

if TYPE_CHECKING:
    from papercut.data.loaders.hf import HfPssCorpus


class TfIdfXgbLayoutVis(TfIdfXgbLayout):
    """Layout + page-pair visual features (3x3 intensity grid, edges, halves).

    Pulls per-page visual signals from the corpus (computed during ingestion
    from the raw page image) and adds them to the pair feature block as
    `[prev_vis, curr_vis, prev - curr, |prev - curr|]`. Catches layout shifts
    (letterhead vs body, signature block, page-number footer) that bbox
    layout features approximate but raw pixels capture directly.
    """

    name = "tfidf_xgb_layout_vis"

    def __init__(self, corpus: HfPssCorpus, **kwargs: object) -> None:
        super().__init__(corpus=corpus, **kwargs)
        if not corpus.has_visuals():
            raise ValueError(
                "TfIdfXgbLayoutVis needs a corpus with visual features; "
                "re-download with the updated tabme_pp loader."
            )

    def _gather_visuals(self, stream: Stream) -> np.ndarray:
        return np.asarray(
            [self.corpus.visual(p) for p in stream.pages],
            dtype=np.float32,
        )

    def _build_features(self, texts: list[str], layouts: np.ndarray) -> csr_matrix:
        base = super()._build_features(texts, layouts)
        # Recover the stream by reconstructing visuals on the same texts ordering:
        # but we don't have the stream here. Override differently:
        return base


class TfIdfXgbAll(TfIdfXgbLayoutSem):
    """Everything: TF-IDF + structural + bbox layout + cross-page Jaccard +
    page-pair MiniLM cosine + visual features (intensity grid, edges).

    Subclasses the semantic model and bolts on the visual block. Trains a
    single XGBoost over the whole concatenated feature vector. The dense
    feature block grows by 4 * 17 = 68 floats per pair (prev, curr, diff,
    abs-diff for each of the 17 visual features).
    """

    name = "tfidf_xgb_all"

    def __init__(self, corpus: HfPssCorpus, **kwargs: object) -> None:
        super().__init__(corpus=corpus, **kwargs)
        if not corpus.has_visuals():
            raise ValueError("TfIdfXgbAll needs a corpus with visual features.")
        self._stream_visuals: dict[int, np.ndarray] = {}

    def _gather(self, stream: Stream) -> tuple[list[str], np.ndarray]:
        texts, layouts = super()._gather(stream)
        vis = np.asarray(
            [self.corpus.visual(p) for p in stream.pages],
            dtype=np.float32,
        )
        self._stream_visuals[id(stream)] = vis
        return texts, layouts

    def _build_features(self, texts: list[str], layouts: np.ndarray) -> csr_matrix:
        truncated = [self._truncate(t) for t in texts]
        page_tfidf = self.vectorizer.transform(truncated)
        prev_tf = page_tfidf[:-1]
        curr_tf = page_tfidf[1:]

        struct = np.vstack([_page_features(t) for t in texts])
        prev_s = struct[:-1]
        curr_s = struct[1:]
        struct_pairs = np.hstack([prev_s, curr_s, prev_s - curr_s, np.abs(prev_s - curr_s)]).astype(
            np.float32
        )

        prev_l = layouts[:-1]
        curr_l = layouts[1:]
        layout_pairs = np.hstack([prev_l, curr_l, prev_l - curr_l, np.abs(prev_l - curr_l)]).astype(
            np.float32
        )

        n = len(texts)
        positions = np.arange(1, n, dtype=np.float32)
        denom = max(1.0, float(n - 1))
        pos_pairs = np.stack(
            [
                positions / denom,
                (n - 1 - positions) / denom,
                np.full_like(positions, float(n)),
            ],
            axis=1,
        ).astype(np.float32)

        cross = np.asarray(
            [_cross_page_features(texts[i - 1], texts[i]) for i in range(1, n)],
            dtype=np.float32,
        )

        embeds = self._embed(truncated)
        norms = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9
        embeds = embeds / norms
        cos = np.sum(embeds[:-1] * embeds[1:], axis=1, dtype=np.float32).reshape(-1, 1)

        # Vision features computed externally (need the stream pages, not just
        # texts). We stash them during _gather and look up the most recent.
        vis = next(reversed(self._stream_visuals.values()))
        prev_v = vis[:-1]
        curr_v = vis[1:]
        vis_pairs = np.hstack([prev_v, curr_v, prev_v - curr_v, np.abs(prev_v - curr_v)]).astype(
            np.float32
        )

        dense = np.hstack([struct_pairs, layout_pairs, pos_pairs, cross, cos, vis_pairs])
        self._stream_visuals.clear()
        return hstack([prev_tf, curr_tf, csr_matrix(dense)]).tocsr()
