from __future__ import annotations

import pickle
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer

from papercut.streams.resolver import PageResolver
from papercut.streams.types import Stream

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _page_features(text: str) -> np.ndarray:
    """Cheap language-agnostic page summary features."""
    if not text:
        return np.zeros(8, dtype=np.float32)
    n = len(text)
    n_log = float(np.log1p(n))
    words = _WORD_RE.findall(text)
    n_words = len(words)
    avg_word = float(n / n_words) if n_words else 0.0
    alpha = sum(c.isalpha() for c in text)
    digit = sum(c.isdigit() for c in text)
    upper = sum(c.isupper() for c in text)
    punct = sum(c in ".,;:!?-/" for c in text)
    first_line = text.splitlines()[0] if "\n" in text else text[:100]
    return np.asarray(
        [
            n_log,
            n_words / max(1.0, n) * 100.0,
            avg_word,
            alpha / max(1, n),
            digit / max(1, n),
            upper / max(1, alpha),
            punct / max(1, n),
            len(first_line) / max(1, n),
        ],
        dtype=np.float32,
    )


def _pair_struct_features(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    return np.concatenate([prev, curr, prev - curr, np.abs(prev - curr)])


class TfIdfXgbRich:
    """TF-IDF + page-level structural features fed into XGBoost.

    Same shape as `TfIdfXgb` but augments the page-pair TF-IDF representation
    with eight cheap structural features per page (length, word stats,
    alpha/digit/upper/punct ratios, first-line length). Boundary pages
    typically carry letterheads, more capitalisation, and longer first
    lines, so these features add complementary signal at near-zero cost.
    """

    name = "tfidf_xgb_rich"

    def __init__(
        self,
        resolver: PageResolver,
        ngram_range: tuple[int, int] = (1, 2),
        analyzer: str = "word",
        max_features: int = 20_000,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        max_chars_per_page: int = 4000,
        random_state: int = 0,
    ) -> None:
        self.resolver = resolver
        self.max_chars_per_page = max_chars_per_page
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            sublinear_tf=True,
        )
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
        )
        self._fitted = False

    def _truncate(self, text: str) -> str:
        half = self.max_chars_per_page // 2
        if len(text) <= self.max_chars_per_page:
            return text
        return text[:half] + " " + text[-half:]

    def _texts(self, stream: Stream) -> list[str]:
        return [self.resolver.text(p) for p in stream.pages]

    def _build_features(self, texts: list[str]) -> csr_matrix:
        truncated = [self._truncate(t) for t in texts]
        page_vecs = self.vectorizer.transform(truncated)
        prev = page_vecs[:-1]
        curr = page_vecs[1:]
        struct = np.vstack([_page_features(t) for t in texts])
        prev_s = struct[:-1]
        curr_s = struct[1:]
        struct_pairs = np.hstack([prev_s, curr_s, prev_s - curr_s, np.abs(prev_s - curr_s)])
        struct_sparse = csr_matrix(struct_pairs.astype(np.float32))
        return hstack([prev, curr, struct_sparse]).tocsr()

    def fit(self, streams: Sequence[Stream]) -> None:
        all_texts: list[str] = []
        per_stream_texts: list[list[str]] = []
        for stream in streams:
            if stream.boundaries is None:
                raise ValueError("Cannot fit on unlabeled stream")
            texts = self._texts(stream)
            per_stream_texts.append(texts)
            all_texts.extend(self._truncate(t) for t in texts)
        if not all_texts:
            raise ValueError("No texts available")
        self.vectorizer.fit(all_texts)

        blocks: list[csr_matrix] = []
        labels: list[int] = []
        for stream, texts in zip(streams, per_stream_texts, strict=True):
            assert stream.boundaries is not None
            if len(texts) < 2:
                continue
            blocks.append(self._build_features(texts))
            labels.extend(1 if b else 0 for b in stream.boundaries[1:])
        if not blocks:
            raise ValueError("Need at least one multi-page stream to fit")
        x_train = vstack(blocks).tocsr()
        y_train = np.asarray(labels, dtype=np.int32)
        self.model.fit(x_train, y_train)
        self._fitted = True

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        if not self._fitted:
            raise RuntimeError("TfIdfXgbRich must be fit before predict_probs")
        texts = self._texts(stream)
        if len(texts) < 2:
            return (1.0,)
        features = self._build_features(texts)
        proba = self.model.predict_proba(features)[:, 1].tolist()
        return (1.0, *proba)

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        probs = self.predict_probs(stream)
        return (True, *(p > 0.5 for p in probs[1:]))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "vectorizer": self.vectorizer,
            "model": self.model,
            "max_chars_per_page": self.max_chars_per_page,
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> Self:
        raise NotImplementedError("Use load_with_resolver")

    @classmethod
    def load_with_resolver(cls, path: str, resolver: PageResolver) -> TfIdfXgbRich:
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls.__new__(cls)
        instance.resolver = resolver
        instance.max_chars_per_page = state["max_chars_per_page"]
        instance.vectorizer = state["vectorizer"]
        instance.model = state["model"]
        instance._fitted = state["fitted"]
        return instance
