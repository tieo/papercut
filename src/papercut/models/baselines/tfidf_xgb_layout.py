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
from papercut.streams.types import Stream


def _char_ngrams(text: str, n: int) -> set[str]:
    text = text.strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _digit_run_count(text: str) -> int:
    """Count maximal runs of 1 to 4 digits. Approximates page numbers and IDs."""
    count = 0
    in_run = False
    run_len = 0
    for c in text:
        if c.isdigit():
            in_run = True
            run_len += 1
        else:
            if in_run and 1 <= run_len <= 4:
                count += 1
            in_run = False
            run_len = 0
    if in_run and 1 <= run_len <= 4:
        count += 1
    return count


def _cross_page_features(prev: str, curr: str, head: int = 300, foot: int = 300) -> list[float]:
    """Language-agnostic similarity signals between consecutive pages.

    Head and tail similarity catch shared letterheads and footers (strong
    same-doc cue). Full-text Jaccard catches body-text overlap. Length
    asymmetry catches the cover-page / continuation-page contrast. The
    extra-narrow head/foot windows (50 chars) emphasise the very-top and
    very-bottom layout regions; digit-run counts in the footer approximate
    page-number presence without depending on a specific language pattern.
    """
    prev_head = prev[:head]
    curr_head = curr[:head]
    prev_foot = prev[-foot:]
    curr_foot = curr[-foot:]
    prev_top = prev[:50]
    curr_top = curr[:50]
    prev_bot = prev[-50:]
    curr_bot = curr[-50:]

    head_sim = _jaccard(_char_ngrams(prev_head, 4), _char_ngrams(curr_head, 4))
    foot_sim = _jaccard(_char_ngrams(prev_foot, 4), _char_ngrams(curr_foot, 4))
    full_sim = _jaccard(_char_ngrams(prev, 4), _char_ngrams(curr, 4))
    top_sim = _jaccard(_char_ngrams(prev_top, 3), _char_ngrams(curr_top, 3))
    bot_sim = _jaccard(_char_ngrams(prev_bot, 3), _char_ngrams(curr_bot, 3))

    prev_words = set(prev_head.lower().split())
    curr_words = set(curr_head.lower().split())
    head_word_sim = _jaccard(prev_words, curr_words)

    prev_len = max(1, len(prev))
    curr_len = max(1, len(curr))
    len_ratio = min(prev_len, curr_len) / max(prev_len, curr_len)
    log_len_diff = abs(np.log1p(prev_len) - np.log1p(curr_len))

    prev_digits = _digit_run_count(prev_foot)
    curr_digits = _digit_run_count(curr_foot)

    return [
        head_sim,
        foot_sim,
        full_sim,
        top_sim,
        bot_sim,
        head_word_sim,
        len_ratio,
        float(log_len_diff),
        float(prev_digits),
        float(curr_digits),
        float(abs(prev_digits - curr_digits)),
    ]


if TYPE_CHECKING:
    from papercut.data.loaders.hf import HfPssCorpus


class TfIdfXgbLayout:
    """TF-IDF + page-structural + OCR-bbox layout features into XGBoost.

    Adds the corpus-provided 14 layout signals (header/footer density,
    address-block region, vertical word spread, mean word height) to the
    rich TF-IDF baseline. Layout features come from the OCR JSON bbox
    coordinates that TABME++ stores per page, so they cost nothing extra
    at inference time and capture letterhead / signature-block shape that
    text alone misses. Closes the gap toward Guha et al. 2022's multimodal
    approach without needing rendered images.
    """

    name = "tfidf_xgb_layout"

    def __init__(
        self,
        corpus: HfPssCorpus,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 20_000,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        max_chars_per_page: int = 4000,
        random_state: int = 0,
    ) -> None:
        self.corpus = corpus
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
        self._n_estimators = n_estimators
        self._fitted = False

    def _truncate(self, text: str) -> str:
        half = self.max_chars_per_page // 2
        if len(text) <= self.max_chars_per_page:
            return text
        return text[:half] + " " + text[-half:]

    def _gather(self, stream: Stream) -> tuple[list[str], np.ndarray]:
        texts: list[str] = []
        layouts: list[list[float]] = []
        for page in stream.pages:
            texts.append(self.corpus.text(page))
            layouts.append(self.corpus.layout(page))
        return texts, np.asarray(layouts, dtype=np.float32)

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

        dense = np.hstack([struct_pairs, layout_pairs, pos_pairs, cross])
        return hstack([prev_tf, curr_tf, csr_matrix(dense)]).tocsr()

    def fit(self, streams: Sequence[Stream]) -> None:
        all_truncated: list[str] = []
        per_stream: list[tuple[list[str], np.ndarray]] = []
        for stream in streams:
            if stream.boundaries is None:
                raise ValueError("Cannot fit on unlabeled stream")
            texts, layouts = self._gather(stream)
            per_stream.append((texts, layouts))
            all_truncated.extend(self._truncate(t) for t in texts)
        if not all_truncated:
            raise ValueError("No texts available")
        self.vectorizer.fit(all_truncated)

        blocks: list[csr_matrix] = []
        labels: list[int] = []
        for stream, (texts, layouts) in zip(streams, per_stream, strict=True):
            assert stream.boundaries is not None
            if len(texts) < 2:
                continue
            blocks.append(self._build_features(texts, layouts))
            labels.extend(1 if b else 0 for b in stream.boundaries[1:])
        if not blocks:
            raise ValueError("Need at least one multi-page stream to fit")
        x_train = vstack(blocks).tocsr()
        y_train = np.asarray(labels, dtype=np.int32)
        self.model.fit(x_train, y_train)
        self._fitted = True

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        if not self._fitted:
            raise RuntimeError("TfIdfXgbLayout must be fit before predict_probs")
        texts, layouts = self._gather(stream)
        if len(texts) < 2:
            return (1.0,)
        features = self._build_features(texts, layouts)
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
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
