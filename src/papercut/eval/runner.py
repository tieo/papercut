from __future__ import annotations

from collections.abc import Iterable
from statistics import mean
from typing import NamedTuple

from papercut.eval.metrics import mndd, page_metrics, panoptic_quality, stp
from papercut.models.base import Model
from papercut.streams.types import Stream


class EvalReport(NamedTuple):
    n_streams: int
    page_f1_mean: float
    pq_mean: float
    stp: float
    mndd_mean: float


def evaluate(model: Model, streams: Iterable[Stream]) -> EvalReport:
    """Run a model over labeled streams and aggregate per-stream metrics.

    Each metric is computed per stream and then averaged across streams, so
    long streams do not dominate short ones. Streams must carry ground-truth
    boundaries.
    """
    streams = list(streams)
    if not streams:
        raise ValueError("evaluate() requires at least one stream")

    pairs: list[tuple[tuple[bool, ...], tuple[bool, ...]]] = []
    page_f1s: list[float] = []
    pqs: list[float] = []
    mndds: list[int] = []

    for stream in streams:
        if stream.boundaries is None:
            raise ValueError(f"Stream is unlabeled: {stream.pages[0]}")
        pred = model.predict_boundaries(stream)
        if len(pred) != len(stream):
            raise ValueError(
                f"Model {model.name!r} returned {len(pred)} predictions "
                f"for a stream of length {len(stream)}"
            )
        pairs.append((stream.boundaries, pred))
        page_f1s.append(page_metrics(stream.boundaries, pred).f1)
        pqs.append(panoptic_quality(stream.boundaries, pred).pq)
        mndds.append(mndd(stream.boundaries, pred))

    return EvalReport(
        n_streams=len(streams),
        page_f1_mean=mean(page_f1s),
        pq_mean=mean(pqs),
        stp=stp(pairs),
        mndd_mean=mean(mndds),
    )
