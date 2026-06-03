from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from papercut.streams.resolver import PageResolver
from papercut.streams.types import Stream

PageEncoder = Callable[[list[str]], np.ndarray]


class MultilingualMiniLM:
    """Page boundary classifier over sentence-transformer embeddings.

    Each page's text gets encoded into a dense vector by a multilingual
    sentence-transformer (default: paraphrase-multilingual-MiniLM-L12-v2,
    ~118M params, 50+ languages, CPU-friendly). Page-pair features are the
    concatenation of (prev, curr, prev*curr, |prev - curr|), fed into a
    logistic regression. The signal is semantic similarity between
    consecutive pages: same-doc pages share topic and vocabulary, new-doc
    pages do not.
    """

    name = "multilingual_minilm"

    def __init__(
        self,
        resolver: PageResolver,
        encoder: PageEncoder | None = None,
        max_chars_per_page: int = 4000,
        c: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.resolver = resolver
        self.max_chars_per_page = max_chars_per_page
        self._encoder = encoder
        self._c = c
        self._random_state = random_state
        self._classifier: object | None = None

    @staticmethod
    def _build_default_encoder() -> PageEncoder:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "MultilingualMiniLM needs the 'ml' extra: `uv sync --extra ml`"
            ) from e
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        st_model = SentenceTransformer(model_name)

        def encode(texts: list[str]) -> np.ndarray:
            arr = st_model.encode(
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

    def _page_texts(self, stream: Stream) -> list[str]:
        return [self._truncate(self.resolver.text(p)) for p in stream.pages]

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0,), dtype=np.float32)
        return self._ensure_encoder()(texts)

    @staticmethod
    def _pair_features(page_vecs: np.ndarray) -> np.ndarray:
        if page_vecs.shape[0] < 2:
            return np.empty((0, 0), dtype=np.float32)
        prev = page_vecs[:-1]
        curr = page_vecs[1:]
        return np.concatenate([prev, curr, prev * curr, np.abs(prev - curr)], axis=1)

    def fit(self, streams: Sequence[Stream]) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as e:
            raise ImportError(
                "MultilingualMiniLM needs the 'classical' extra: `uv sync --extra classical`"
            ) from e

        feature_blocks: list[np.ndarray] = []
        labels: list[int] = []
        for stream in streams:
            if stream.boundaries is None:
                raise ValueError("Cannot fit on unlabeled stream")
            texts = self._page_texts(stream)
            if len(texts) < 2:
                continue
            vecs = self._embed(texts)
            feature_blocks.append(self._pair_features(vecs))
            labels.extend(1 if b else 0 for b in stream.boundaries[1:])
        if not feature_blocks:
            raise ValueError("Need at least one multi-page stream to fit")
        x_train = np.vstack(feature_blocks)
        y_train = np.asarray(labels, dtype=np.int32)
        clf = LogisticRegression(
            C=self._c,
            max_iter=1000,
            random_state=self._random_state,
            n_jobs=-1,
        )
        clf.fit(x_train, y_train)
        self._classifier = clf

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        if self._classifier is None:
            raise RuntimeError("MultilingualMiniLM must be fit before predict_probs")
        texts = self._page_texts(stream)
        if len(texts) < 2:
            return (1.0,)
        vecs = self._embed(texts)
        features = self._pair_features(vecs)
        proba = self._classifier.predict_proba(features)[:, 1].tolist()  # type: ignore[attr-defined]
        return (1.0, *proba)

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        probs = self.predict_probs(stream)
        return (True, *(p > 0.5 for p in probs[1:]))

    def save(self, path: str | Path) -> None:
        if self._classifier is None:
            raise RuntimeError("Cannot save unfitted model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "classifier": self._classifier,
            "max_chars_per_page": self.max_chars_per_page,
            "c": self._c,
            "random_state": self._random_state,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_with_resolver(
        cls,
        path: str | Path,
        resolver: PageResolver,
        encoder: PageEncoder | None = None,
    ) -> MultilingualMiniLM:
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls(
            resolver=resolver,
            encoder=encoder,
            max_chars_per_page=state["max_chars_per_page"],
            c=state["c"],
            random_state=state["random_state"],
        )
        instance._classifier = state["classifier"]
        return instance
