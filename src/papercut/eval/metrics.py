from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple


class PRF(NamedTuple):
    precision: float
    recall: float
    f1: float


class PanopticQuality(NamedTuple):
    """Document-level segmentation quality (Kirillov et al. 2019).

    RQ is the document-level F1 over (pred, true) document pairs with
    IoU > 0.5. SQ is the mean IoU of matched pairs. PQ = RQ * SQ.
    """

    rq: float
    sq: float
    pq: float


def page_metrics(true: Sequence[bool], pred: Sequence[bool]) -> PRF:
    """Per-page boundary precision / recall / F1."""
    raise NotImplementedError


def panoptic_quality(true: Sequence[bool], pred: Sequence[bool]) -> PanopticQuality:
    """Document-level RQ / SQ / PQ (IoU > 0.5 matching)."""
    raise NotImplementedError


def stp(pairs: Sequence[tuple[Sequence[bool], Sequence[bool]]]) -> float:
    """Straight-through processing: fraction of streams predicted exactly."""
    raise NotImplementedError


def mndd(true: Sequence[bool], pred: Sequence[bool]) -> int:
    """Minimum number of page drag-and-drops to fix prediction.

    From Mungmeeprued et al. 2022. Reflects user-visible repair effort.
    """
    raise NotImplementedError
