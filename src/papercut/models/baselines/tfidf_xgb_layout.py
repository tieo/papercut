from __future__ import annotations

import pickle
import re
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


_PAGINATION = re.compile(r"\b(\d{1,3})\s*(?:[/|-]|[^\W\d_]{1,12}\s)\s*(\d{1,3})\b")


def _pagination(text: str, window: int = 400) -> tuple[bool, int, int]:
    """Read a "page k of n" mark off the head and foot of a page.

    Printed pagination is the one place a document states its own extent, and
    an invoice run from one sender is where every other cue fails: the
    letterhead, the layout and the wording repeat, while the count restarts.
    The pattern is numeric, a pair of small numbers joined by a separator or a
    short word, so it reads "1/2", "1 of 2" and "Seite 1 von 2" alike without
    naming a language. The last plausible pair wins, since a page number sits
    at the end of a header or footer line.
    """
    best: tuple[int, int] | None = None
    for part in (text[:window], text[-window:]):
        for match in _PAGINATION.finditer(part):
            index, total = int(match.group(1)), int(match.group(2))
            if 1 <= index <= total <= 99:
                best = (index, total)
    if best is None:
        return False, 0, 0
    return True, best[0], best[1]


def _pagination_pair_features(prev: str, curr: str) -> list[float]:
    """Compare the pagination of two pages, zeros when either lacks one."""
    prev_found, prev_index, prev_total = _pagination(prev)
    curr_found, curr_index, curr_total = _pagination(curr)
    if not (prev_found and curr_found):
        return [float(prev_found), float(curr_found), 0.0, 0.0, 0.0, 0.0]
    return [
        float(prev_found),
        float(curr_found),
        float(prev_total == curr_total),
        float(curr_index == prev_index + 1 and prev_total == curr_total),
        float(curr_index == 1 and prev_index > 1),
        float(prev_index == prev_total and curr_index == 1),
    ]


def _cross_page_features(
    prev: str, curr: str, head: int = 300, foot: int = 300, pagination: bool = False
) -> list[float]:
    """Language-agnostic similarity signals between consecutive pages.

    Multi-scale head and tail similarity catches shared letterheads and
    footers at varying window widths (strong same-doc cue). Full-text
    Jaccard catches body-text overlap. Length asymmetry catches the
    cover-page / continuation-page contrast. Digit-run counts in the footer
    approximate page-number presence without depending on a specific
    language pattern.
    """

    def head_chars(text: str, n: int) -> str:
        return text[:n]

    def foot_chars(text: str, n: int) -> str:
        return text[-n:]

    # Multi-scale head and foot Jaccard
    multi_head: list[float] = []
    multi_foot: list[float] = []
    for window in (50, 150, 300, 600):
        ph = head_chars(prev, window)
        ch = head_chars(curr, window)
        pf = foot_chars(prev, window)
        cf = foot_chars(curr, window)
        multi_head.append(_jaccard(_char_ngrams(ph, 4), _char_ngrams(ch, 4)))
        multi_foot.append(_jaccard(_char_ngrams(pf, 4), _char_ngrams(cf, 4)))

    full_sim = _jaccard(_char_ngrams(prev, 4), _char_ngrams(curr, 4))
    top_sim = _jaccard(_char_ngrams(prev[:50], 3), _char_ngrams(curr[:50], 3))
    bot_sim = _jaccard(_char_ngrams(prev[-50:], 3), _char_ngrams(curr[-50:], 3))

    prev_words = set(prev[:head].lower().split())
    curr_words = set(curr[:head].lower().split())
    head_word_sim = _jaccard(prev_words, curr_words)

    prev_len = max(1, len(prev))
    curr_len = max(1, len(curr))
    len_ratio = min(prev_len, curr_len) / max(prev_len, curr_len)
    log_len_diff = abs(np.log1p(prev_len) - np.log1p(curr_len))

    prev_digits = _digit_run_count(foot_chars(prev, foot))
    curr_digits = _digit_run_count(foot_chars(curr, foot))

    extra = _pagination_pair_features(prev, curr) if pagination else []
    return [
        *multi_head,
        *multi_foot,
        full_sim,
        top_sim,
        bot_sim,
        head_word_sim,
        len_ratio,
        float(log_len_diff),
        float(prev_digits),
        float(curr_digits),
        float(abs(prev_digits - curr_digits)),
        *extra,
    ]


def standardise_within_stream(block: np.ndarray) -> np.ndarray:
    """Restate each column as standard deviations from this stream's mean.

    Absolute similarity carries a scale that belongs to the corpus a model was
    fitted on: a text overlap of 0.4 separates documents in a stack of dense
    letters and holds one together in a stack of sparse forms, and a model
    taught the absolute number learns the wrong rule for the second stack.
    The standing of a pair among its own stream's pairs survives that change,
    since it is measured against the stack it came from.

    A stream of one pair has nothing to compare against and comes back as
    zeros, which says exactly that.
    """
    if block.shape[0] < 2:
        return np.zeros_like(block, dtype=np.float32)
    mean = block.mean(axis=0, keepdims=True)
    deviation = block.std(axis=0, keepdims=True)
    return ((block - mean) / (deviation + 1e-6)).astype(np.float32)


def _stream_context_features(cross: np.ndarray) -> np.ndarray:
    """Place each page pair against the rest of its stream.

    A whole scan is available at once, so a pair is judged against its
    neighbours rather than on its own: the similarity that marks a document
    start in a stack of dense reports can be ordinary inside a stack of
    sparse forms. Each pair carries the preceding and following pair, the
    difference to both, and its distance from the stream mean in standard
    deviations, which is what makes a dip in similarity readable as a
    boundary.
    """
    if cross.size == 0:
        return np.zeros((cross.shape[0], 0), dtype=np.float32)
    previous = np.vstack([cross[:1], cross[:-1]])
    following = np.vstack([cross[1:], cross[-1:]])
    mean = cross.mean(axis=0, keepdims=True)
    deviation = cross.std(axis=0, keepdims=True) + 1e-6
    return np.hstack(
        [
            previous,
            following,
            cross - previous,
            cross - following,
            (cross - mean) / deviation,
        ]
    ).astype(np.float32)


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
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        colsample_bytree: float = 1.0,
        max_bin: int = 256,
        max_chars_per_page: int = 4000,
        threshold: float = 0.5,
        random_state: int = 0,
        analyzer: str = "word",
        context_features: bool = True,
        pagination_features: bool = True,
        standardised_features: bool = False,
    ) -> None:
        self.corpus = corpus
        self.max_chars_per_page = max_chars_per_page
        self.threshold = threshold
        self.context_features = context_features
        self.pagination_features = pagination_features
        self.standardised_features = standardised_features
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
            colsample_bytree=colsample_bytree,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            max_bin=max_bin,
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

        prev_l_norm = np.linalg.norm(prev_l, axis=1, keepdims=True) + 1e-9
        curr_l_norm = np.linalg.norm(curr_l, axis=1, keepdims=True) + 1e-9
        layout_cos = (
            np.sum(prev_l * curr_l, axis=1, keepdims=True) / (prev_l_norm * curr_l_norm)
        ).astype(np.float32)

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
            [
                _cross_page_features(
                    texts[i - 1], texts[i], pagination=self.pagination_features
                )
                for i in range(1, n)
            ],
            dtype=np.float32,
        )

        blocks = [struct_pairs, layout_pairs, layout_cos, pos_pairs, cross]
        if self.context_features:
            blocks.append(_stream_context_features(cross))
        dense = np.hstack(blocks)
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
        return (True, *(p > self.threshold for p in probs[1:]))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "vectorizer": self.vectorizer,
            "model": self.model,
            "max_chars_per_page": self.max_chars_per_page,
            "threshold": self.threshold,
            "context_features": self.context_features,
            "pagination_features": self.pagination_features,
            "standardised_features": self.standardised_features,
            "model_class": "TfIdfXgbLayout",
        }
        with target.open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_with_corpus(cls, path: str | Path, corpus: HfPssCorpus) -> TfIdfXgbLayout:
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        instance = cls.__new__(cls)
        instance.corpus = corpus
        instance.vectorizer = state["vectorizer"]
        instance.model = state["model"]
        instance.max_chars_per_page = state["max_chars_per_page"]
        instance.threshold = state.get("threshold", 0.5)
        # Models fitted before stream context existed carry the narrower
        # feature layout, so the absent key means those columns stay off.
        instance.context_features = state.get("context_features", False)
        instance.pagination_features = state.get("pagination_features", False)
        instance.standardised_features = state.get("standardised_features", False)
        instance._fitted = True
        return instance
