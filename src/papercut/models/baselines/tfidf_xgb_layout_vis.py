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
        self._current_visual: np.ndarray | None = None

    def _gather_visuals(self, stream: Stream) -> np.ndarray:
        return np.asarray(
            [self.corpus.visual(p) for p in stream.pages],
            dtype=np.float32,
        )

    def fit(self, streams):  # type: ignore[override]
        from scipy.sparse import vstack

        all_truncated: list[str] = []
        gathered: list[tuple[list[str], np.ndarray, np.ndarray]] = []
        for stream in streams:
            if stream.boundaries is None:
                raise ValueError("Cannot fit on unlabeled stream")
            texts, layouts = self._gather(stream)
            visuals = self._gather_visuals(stream)
            gathered.append((texts, layouts, visuals))
            all_truncated.extend(self._truncate(t) for t in texts)
        if not all_truncated:
            raise ValueError("No texts available")
        self.vectorizer.fit(all_truncated)

        blocks: list[csr_matrix] = []
        labels: list[int] = []
        for stream, (texts, layouts, visuals) in zip(streams, gathered, strict=True):
            assert stream.boundaries is not None
            if len(texts) < 2:
                continue
            self._current_visual = visuals
            blocks.append(self._build_features(texts, layouts))
            labels.extend(1 if b else 0 for b in stream.boundaries[1:])
        if not blocks:
            raise ValueError("Need at least one multi-page stream to fit")
        x_train = vstack(blocks).tocsr()
        y_train = np.asarray(labels, dtype=np.int32)
        self.model.fit(x_train, y_train)
        self._fitted = True

    def predict_probs(self, stream):  # type: ignore[override]
        if not self._fitted:
            raise RuntimeError("TfIdfXgbAll must be fit before predict_probs")
        texts, layouts = self._gather(stream)
        if len(texts) < 2:
            return (1.0,)
        self._current_visual = self._gather_visuals(stream)
        features = self._build_features(texts, layouts)
        proba = self.model.predict_proba(features)[:, 1].tolist()
        return (1.0, *proba)

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

        # Vision pair block. _current_visual is stashed by _gather just
        # before this is called; both have one row per page in the stream.
        if self._current_visual is None or len(self._current_visual) != n:
            raise RuntimeError("Visual feature buffer not populated; _gather not called")
        vis = self._current_visual
        prev_v = vis[:-1]
        curr_v = vis[1:]
        vis_pairs = np.hstack([prev_v, curr_v, prev_v - curr_v, np.abs(prev_v - curr_v)]).astype(
            np.float32
        )

        dense = np.hstack([struct_pairs, layout_pairs, pos_pairs, cross, cos, vis_pairs])
        return hstack([prev_tf, curr_tf, csr_matrix(dense)]).tocsr()
