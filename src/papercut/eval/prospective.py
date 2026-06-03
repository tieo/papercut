from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from papercut.eval.runner import EvalReport, evaluate
from papercut.models.base import Model
from papercut.streams.types import Stream


@dataclass(frozen=True)
class Slice:
    """A time-ordered chunk of labeled streams.

    Slices represent successive snapshots of data. The convention is that
    `slices[i]` was collected before `slices[i+1]`, so walk-forward training
    always uses earlier slices for training and later ones for evaluation.
    """

    name: str
    streams: list[Stream]


class SliceResult(NamedTuple):
    model: str
    train_slices: tuple[str, ...]
    test_slice: str
    report: EvalReport


def walk_forward(
    models: Sequence[Model],
    slices: Sequence[Slice],
) -> list[SliceResult]:
    """For each i in [0, len(slices) - 1), train then evaluate.

    Trainable models are fit on the concatenation of slices[0..i]; all models
    are evaluated on slices[i+1]. Non-trainable models (no `fit` method) are
    evaluated as-is on each test slice, which yields the fair baseline
    comparison: zero-shot models see every test slice as "new" because they
    never trained on anything.
    """
    if len(slices) < 2:
        raise ValueError("walk_forward needs at least 2 slices")

    results: list[SliceResult] = []
    for i in range(len(slices) - 1):
        train_streams: list[Stream] = []
        for s in slices[: i + 1]:
            train_streams.extend(s.streams)
        test_slice = slices[i + 1]
        train_names = tuple(s.name for s in slices[: i + 1])

        for model in models:
            if callable(getattr(model, "fit", None)):
                model.fit(train_streams)  # type: ignore[attr-defined]
            report = evaluate(model, test_slice.streams)
            results.append(
                SliceResult(
                    model=model.name,
                    train_slices=train_names,
                    test_slice=test_slice.name,
                    report=report,
                )
            )
    return results


def format_results(results: Sequence[SliceResult]) -> str:
    """Render results as a simple aligned table for stdout."""
    header = f"{'model':24s} {'test slice':18s} {'page_f1':>8s} {'pq':>6s} {'stp':>6s} {'mndd':>6s}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.model:24s} {r.test_slice:18s} "
            f"{r.report.page_f1_mean:>8.3f} "
            f"{r.report.pq_mean:>6.3f} "
            f"{r.report.stp:>6.3f} "
            f"{r.report.mndd_mean:>6.2f}"
        )
    return "\n".join(lines)
