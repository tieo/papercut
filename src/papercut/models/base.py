from __future__ import annotations

from typing import Protocol, runtime_checkable

from papercut.streams.types import Stream


@runtime_checkable
class Model(Protocol):
    """Anything that can predict document boundaries for a stream of pages.

    All architectures (XGBoost, DiT, LayoutXLM, ...) implement this same
    surface so the evaluation harness can run any model on any slice.
    """

    name: str

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        """Return a boundary vector aligned to `stream.pages`."""
        ...


@runtime_checkable
class ProbabilisticModel(Model, Protocol):
    """A `Model` that also emits per-page boundary probabilities.

    The probability at position 0 is always 1.0 by convention (page 0 starts
    a document by definition). Probabilities are needed for late ensembling
    and for sequence-smoothing decoders (Viterbi, CRF).
    """

    def predict_probs(self, stream: Stream) -> tuple[float, ...]: ...


@runtime_checkable
class TrainableModel(Model, Protocol):
    """A `Model` that can be fit on labeled streams."""

    def fit(self, streams: list[Stream]) -> None: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> TrainableModel: ...
