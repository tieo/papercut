from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from papercut.models.base import ProbabilisticModel
from papercut.streams.types import Stream


@dataclass
class SequenceLrSmoothed:
    """Re-classify each page-pair using its prob + nearby probs as features.

    Wraps a probabilistic submodel and trains a logistic regression on a
    window-of-probabilities feature: for each position i, take
    [p_{i-2}, p_{i-1}, p_i, p_{i+1}, p_{i+2}] (boundary at edges) plus the
    stream-level max/mean. Captures sequence-wide patterns the per-pair
    XGBoost cannot see directly without blowing up the feature count.
    """

    submodel: ProbabilisticModel
    window: int = 2
    name: str = field(default="seq_lr_smoothed")

    def __post_init__(self) -> None:
        self._lr: LogisticRegression | None = None

    def _stream_features(self, probs: Sequence[float]) -> np.ndarray:
        n = len(probs)
        if n < 2:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.asarray(probs, dtype=np.float32)
        rows: list[list[float]] = []
        tail_max = float(arr[1:].max()) if n > 1 else 0.0
        tail_mean = float(arr[1:].mean()) if n > 1 else 0.0
        tail_std = float(arr[1:].std()) if n > 1 else 0.0
        for i in range(1, n):
            window = []
            for off in range(-self.window, self.window + 1):
                j = i + off
                if 0 <= j < n:
                    window.append(arr[j])
                else:
                    window.append(0.0)
            rows.append(
                [
                    *window,
                    tail_max,
                    tail_mean,
                    tail_std,
                    float(n),
                    float(i) / max(1, n - 1),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    def fit(self, streams: Sequence[Stream]) -> None:
        if callable(getattr(self.submodel, "fit", None)):
            self.submodel.fit(streams)  # type: ignore[attr-defined]

        feats: list[np.ndarray] = []
        labels: list[int] = []
        for stream in streams:
            if stream.boundaries is None:
                continue
            probs = self.submodel.predict_probs(stream)
            x = self._stream_features(probs)
            if len(x) == 0:
                continue
            feats.append(x)
            labels.extend(1 if b else 0 for b in stream.boundaries[1:])
        if not feats:
            raise ValueError("No training data for sequence smoother")
        X = np.vstack(feats)
        y = np.asarray(labels, dtype=np.int32)
        self._lr = LogisticRegression(max_iter=2000, C=1.0)
        self._lr.fit(X, y)

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        if self._lr is None:
            raise RuntimeError("SequenceLrSmoothed must be fit before predict_probs")
        probs = self.submodel.predict_probs(stream)
        if len(probs) < 2:
            return (1.0,)
        x = self._stream_features(probs)
        out = self._lr.predict_proba(x)[:, 1].tolist()
        return (1.0, *out)

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        probs = self.predict_probs(stream)
        return (True, *(p > 0.5 for p in probs[1:]))
