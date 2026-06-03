from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from papercut.models.base import ProbabilisticModel
from papercut.streams.types import Stream


@dataclass
class LateEnsemble:
    """Average per-page boundary probabilities across submodels.

    Each submodel must satisfy `ProbabilisticModel`. `weights` are normalised;
    by default each model contributes equally. The threshold gates the final
    boolean decision and defaults to 0.5.
    """

    submodels: Sequence[ProbabilisticModel]
    weights: Sequence[float] | None = None
    threshold: float = 0.5
    name: str = field(default="late_ensemble")

    def __post_init__(self) -> None:
        if not self.submodels:
            raise ValueError("LateEnsemble needs at least one submodel")
        if self.weights is None:
            object.__setattr__(self, "weights", [1.0] * len(self.submodels))
        if len(self.weights) != len(self.submodels):
            raise ValueError("weights length must match submodels length")
        if any(w < 0 for w in self.weights):
            raise ValueError("weights must be non-negative")
        if sum(self.weights) == 0:
            raise ValueError("weights cannot all be zero")

    def fit(self, streams: list[Stream]) -> None:
        """Fit any trainable submodel; non-trainable submodels are skipped."""
        for sm in self.submodels:
            if callable(getattr(sm, "fit", None)):
                sm.fit(streams)  # type: ignore[attr-defined]

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        assert self.weights is not None
        n = len(stream)
        agg = [0.0] * n
        total = float(sum(self.weights))
        for sm, w in zip(self.submodels, self.weights, strict=True):
            probs = sm.predict_probs(stream)
            if len(probs) != n:
                raise ValueError(
                    f"submodel {getattr(sm, 'name', type(sm).__name__)!r} returned "
                    f"{len(probs)} probs for a stream of length {n}"
                )
            for i, p in enumerate(probs):
                agg[i] += p * w
        agg = [p / total for p in agg]
        agg[0] = 1.0
        return tuple(agg)

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        probs = self.predict_probs(stream)
        return (True, *(p > self.threshold for p in probs[1:]))
