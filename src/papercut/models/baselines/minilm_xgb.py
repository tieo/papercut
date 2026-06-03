from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer

from papercut.models.baselines.tfidf_xgb_rich import _page_features
from papercut.streams.resolver import PageResolver
from papercut.streams.types import Stream

if TYPE_CHECKING:
    from collections.abc import Callable

    PageEncoder = Callable[[list[str]], np.ndarray]
else:
    PageEncoder = object


class MiniLMXgb:
    """All three text signals fed into one XGBoost classifier.

    Per page:
      - TF-IDF word 1-2 gram sparse vector (high-dim)
      - Multilingual MiniLM dense embedding (384 dim)
      - 8 structural features (length, ratios, first-line)

    Page-pair feature is [prev, curr, prev*curr, |prev - curr|] for each of
    the three blocks, horizontally stacked. XGBoost predicts boundary. Loads
    the encoder lazily so other parts of the project that import this module
    do not pay torch's startup cost.
    """

    name = "minilm_xgb"

    def __init__(
        self,
        resolver: PageResolver,
        encoder: PageEncoder | None = None,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 10_000,
        max_chars_per_page: int = 4000,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        random_state: int = 0,
    ) -> None:
        self.resolver = resolver
        self._encoder = encoder
        self.max_chars_per_page = max_chars_per_page
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
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

    @staticmethod
    def _build_default_encoder() -> PageEncoder:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("MiniLMXgb needs the 'ml' extra: `uv sync --extra ml`") from e
        st = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        def encode(texts: list[str]) -> np.ndarray:
            arr = st.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=16,
            )
            return np.asarray(arr, dtype=np.float32)

        return encode

    def _ensure_encoder(self) -> PageEncoder:
        if self._encoder is None:
            self._encoder = self._build_default_encoder()
        return self._encoder

    def _truncate(self, text: str) -> str:
        half = self.max_chars_per_page // 2
        if len(text) <= self.max_chars_per_page:
            return text
        return text[:half] + " " + text[-half:]

    def _build_features(self, texts: list[str]) -> csr_matrix:
        truncated = [self._truncate(t) for t in texts]
        page_tfidf = self.vectorizer.transform(truncated)
        prev_tf = page_tfidf[:-1]
        curr_tf = page_tfidf[1:]

        embed = self._ensure_encoder()(truncated)
        prev_e = embed[:-1]
        curr_e = embed[1:]
        embed_pairs = np.hstack([prev_e, curr_e, prev_e * curr_e, np.abs(prev_e - curr_e)]).astype(
            np.float32
        )

        struct = np.vstack([_page_features(t) for t in texts])
        prev_s = struct[:-1]
        curr_s = struct[1:]
        struct_pairs = np.hstack([prev_s, curr_s, prev_s - curr_s, np.abs(prev_s - curr_s)]).astype(
            np.float32
        )

        dense_block = np.hstack([embed_pairs, struct_pairs])
        return hstack([prev_tf, curr_tf, csr_matrix(dense_block)]).tocsr()

    def _page_texts(self, stream: Stream) -> list[str]:
        return [self.resolver.text(p) for p in stream.pages]

    def fit(self, streams: Sequence[Stream]) -> None:
        all_truncated: list[str] = []
        per_stream_texts: list[list[str]] = []
        for stream in streams:
            if stream.boundaries is None:
                raise ValueError("Cannot fit on unlabeled stream")
            texts = self._page_texts(stream)
            per_stream_texts.append(texts)
            all_truncated.extend(self._truncate(t) for t in texts)
        if not all_truncated:
            raise ValueError("No texts available")
        self.vectorizer.fit(all_truncated)

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
        x = vstack(blocks).tocsr()
        y = np.asarray(labels, dtype=np.int32)
        self.model.fit(x, y)
        self._fitted = True

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        if not self._fitted:
            raise RuntimeError("MiniLMXgb must be fit before predict_probs")
        texts = self._page_texts(stream)
        if len(texts) < 2:
            return (1.0,)
        features = self._build_features(texts)
        proba = self.model.predict_proba(features)[:, 1].tolist()
        return (1.0, *proba)

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        probs = self.predict_probs(stream)
        return (True, *(p > 0.5 for p in probs[1:]))

    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "vectorizer": self.vectorizer,
            "model": self.model,
            "max_chars_per_page": self.max_chars_per_page,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_with_resolver(
        cls,
        path: str,
        resolver: PageResolver,
        encoder: PageEncoder | None = None,
    ) -> MiniLMXgb:
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls.__new__(cls)
        instance.resolver = resolver
        instance._encoder = encoder
        instance.max_chars_per_page = state["max_chars_per_page"]
        instance.vectorizer = state["vectorizer"]
        instance.model = state["model"]
        instance._fitted = True
        return instance
