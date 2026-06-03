from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer

from papercut.streams.resolver import PageResolver
from papercut.streams.types import Stream


class TfIdfXgb:
    """Page-pair TF-IDF features fed into an XGBoost classifier.

    For each page i greater than zero, the input feature is the horizontal
    concatenation of TF-IDF vectors for pages i-1 and i (character n-grams).
    XGBoost predicts whether page i begins a new document. Page 0 is always
    True by convention and bypasses the model. Matches the strong simple
    baselines reported in the OpenPSS and TABME++ papers.
    """

    name: str = "tfidf_xgb"

    def __init__(
        self,
        resolver: PageResolver,
        char_ngram_range: tuple[int, int] = (3, 5),
        max_features: int = 50_000,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 0,
    ) -> None:
        self.resolver = resolver
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            max_features=max_features,
            lowercase=False,
            sublinear_tf=True,
        )
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=-1,
            eval_metric="logloss",
            random_state=random_state,
        )
        self._fitted = False

    def _texts(self, stream: Stream) -> list[str]:
        return [self.resolver.text(p) for p in stream.pages]

    def _pair_features(self, page_vecs: csr_matrix) -> csr_matrix:
        """Build (n-1, 2*d) sparse matrix of (prev, curr) feature pairs."""
        prev = page_vecs[:-1]
        curr = page_vecs[1:]
        return hstack([prev, curr]).tocsr()

    def fit(self, streams: Sequence[Stream]) -> None:
        all_texts: list[str] = []
        per_stream_texts: list[list[str]] = []
        for stream in streams:
            if stream.boundaries is None:
                raise ValueError("Cannot fit on unlabeled stream")
            texts = self._texts(stream)
            per_stream_texts.append(texts)
            all_texts.extend(texts)
        if not all_texts:
            raise ValueError("No texts available for fitting")
        self.vectorizer.fit(all_texts)

        feature_blocks: list[csr_matrix] = []
        labels: list[int] = []
        for stream, texts in zip(streams, per_stream_texts, strict=True):
            assert stream.boundaries is not None
            if len(texts) < 2:
                continue
            page_vecs = self.vectorizer.transform(texts)
            feature_blocks.append(self._pair_features(page_vecs))
            labels.extend(1 if b else 0 for b in stream.boundaries[1:])

        if not feature_blocks:
            raise ValueError("Need at least one multi-page stream to fit XGBoost")

        x_train = vstack(feature_blocks).tocsr()
        y_train = np.asarray(labels, dtype=np.int32)
        self.model.fit(x_train, y_train)
        self._fitted = True

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        if not self._fitted:
            raise RuntimeError("TfIdfXgb must be fit before predict_boundaries")
        texts = self._texts(stream)
        if len(texts) < 2:
            return (True,)
        page_vecs = self.vectorizer.transform(texts)
        pair_features = self._pair_features(page_vecs)
        preds = self.model.predict(pair_features).astype(bool).tolist()
        return (True, *preds)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "vectorizer": self.vectorizer,
            "model": self.model,
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> Self:
        raise NotImplementedError(
            "Load needs a PageResolver to reattach; use load_with_resolver instead."
        )

    @classmethod
    def load_with_resolver(cls, path: str, resolver: PageResolver) -> TfIdfXgb:
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls.__new__(cls)
        instance.resolver = resolver
        instance.vectorizer = state["vectorizer"]
        instance.model = state["model"]
        instance._fitted = state["fitted"]
        return instance
