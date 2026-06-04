from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, hstack

from papercut.models.baselines.tfidf_xgb_layout import (
    TfIdfXgbLayout,
    _cross_page_features,
)
from papercut.models.baselines.tfidf_xgb_rich import _page_features

if TYPE_CHECKING:
    from collections.abc import Callable

    from papercut.data.loaders.hf import HfPssCorpus

    PageEncoder = Callable[[list[str]], np.ndarray]
else:
    PageEncoder = object


class TfIdfXgbLayoutSem(TfIdfXgbLayout):
    """Layout model plus a single MiniLM cosine-similarity scalar per page-pair.

    Same big bag of features as `TfIdfXgbLayout` (TF-IDF, structural, OCR
    bbox layout, position, 11 cross-page Jaccard signals) but adds one more
    pair feature: cosine similarity between the MiniLM embeddings of the
    prev and curr page. The embedder is loaded lazily (needs the `ml` extra
    at runtime), texts are truncated to keep memory bounded, and an in-
    memory cache by truncated-text-hash means each unique page is encoded
    at most once across fit and predict.
    """

    name = "tfidf_xgb_layout_sem"

    def __init__(
        self,
        corpus: HfPssCorpus,
        encoder: PageEncoder | None = None,
        **layout_kwargs: object,
    ) -> None:
        super().__init__(corpus=corpus, **layout_kwargs)
        self._encoder = encoder
        self._embed_cache: dict[int, np.ndarray] = {}

    def _ensure_encoder(self) -> PageEncoder:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "TfIdfXgbLayoutSem needs the 'ml' extra: `uv sync --extra ml`"
                ) from e
            st = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

            def encode(texts: list[str]) -> np.ndarray:
                arr = st.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=8,
                )
                return np.asarray(arr, dtype=np.float32)

            self._encoder = encode
        return self._encoder

    def _embed(self, texts: list[str]) -> np.ndarray:
        keys = [hash(t) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._embed_cache]
        if missing_idx:
            new_texts = [texts[i] for i in missing_idx]
            new_vecs = self._ensure_encoder()(new_texts)
            for j, i in enumerate(missing_idx):
                self._embed_cache[keys[i]] = new_vecs[j]
        return np.vstack([self._embed_cache[k] for k in keys])

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

        dense = np.hstack([struct_pairs, layout_pairs, pos_pairs, cross, cos])
        return hstack([prev_tf, curr_tf, csr_matrix(dense)]).tocsr()
