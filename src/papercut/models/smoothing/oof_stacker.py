from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from papercut.models.base import ProbabilisticModel
from papercut.streams.types import Stream


@dataclass
class OofStacked:
    """K-fold out-of-fold stacker over a probabilistic submodel.

    Splits train streams into `k_folds`, trains the submodel on K-1 folds,
    predicts on the held-out fold. The OOF predictions provide unbiased
    meta-features (each stream's prediction comes from a model that did not
    see it). A logistic regression learns the meta-classifier on top of the
    same window-of-probs features as `SequenceLrSmoothed`. Finally, the
    submodel is refit on the full training set for inference.
    """

    submodel: ProbabilisticModel
    k_folds: int = 3
    window: int = 2
    seed: int = 0
    name: str = field(default="oof_stacked")

    def __post_init__(self) -> None:
        self._lr: LogisticRegression | None = None

    def _stream_features(self, probs: Sequence[float]) -> np.ndarray:
        n = len(probs)
        if n < 2:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.asarray(probs, dtype=np.float32)
        tail = arr[1:]
        tail_max = float(tail.max()) if len(tail) else 0.0
        tail_mean = float(tail.mean()) if len(tail) else 0.0
        tail_std = float(tail.std()) if len(tail) else 0.0
        rows: list[list[float]] = []
        for i in range(1, n):
            window = []
            for off in range(-self.window, self.window + 1):
                j = i + off
                window.append(arr[j] if 0 <= j < n else 0.0)
            rows.append(
                [*window, tail_max, tail_mean, tail_std, float(n), float(i) / max(1, n - 1)]
            )
        return np.asarray(rows, dtype=np.float32)

    def fit(self, streams: Sequence[Stream]) -> None:
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(streams))
        rng.shuffle(idx)
        folds = np.array_split(idx, self.k_folds)

        feats: list[np.ndarray] = []
        labels: list[int] = []
        for fold_i, fold_idx in enumerate(folds):
            print(f"  OOF fold {fold_i + 1}/{self.k_folds}: fitting submodel...")
            train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fold_i])
            train_streams = [streams[i] for i in train_idx]
            held = [streams[i] for i in fold_idx]
            fold_model = copy.copy(self.submodel)
            # Reset XGBoost / sklearn submodel state. The deepcopy issue
            # makes a fresh tree, but the fitted attributes can leak; we
            # rely on the submodel's fit() being idempotent in practice.
            if hasattr(fold_model, "_fitted"):
                fold_model._fitted = False  # type: ignore[attr-defined]
            fold_model.fit(train_streams)  # type: ignore[attr-defined]
            for s in held:
                if s.boundaries is None:
                    continue
                probs = fold_model.predict_probs(s)
                x = self._stream_features(probs)
                if len(x) == 0:
                    continue
                feats.append(x)
                labels.extend(1 if b else 0 for b in s.boundaries[1:])

        if not feats:
            raise ValueError("No OOF training data")
        X = np.vstack(feats)
        y = np.asarray(labels, dtype=np.int32)
        self._lr = LogisticRegression(max_iter=2000, C=1.0)
        self._lr.fit(X, y)

        print("  fitting final submodel on full train...")
        if hasattr(self.submodel, "_fitted"):
            self.submodel._fitted = False  # type: ignore[attr-defined]
        self.submodel.fit(streams)  # type: ignore[attr-defined]

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        if self._lr is None:
            raise RuntimeError("OofStacked must be fit first")
        probs = self.submodel.predict_probs(stream)
        if len(probs) < 2:
            return (1.0,)
        x = self._stream_features(probs)
        out = self._lr.predict_proba(x)[:, 1].tolist()
        return (1.0, *out)

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        probs = self.predict_probs(stream)
        return (True, *(p > 0.5 for p in probs[1:]))
