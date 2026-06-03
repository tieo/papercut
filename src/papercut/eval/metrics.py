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
    IoU greater than 0.5. SQ is the mean IoU of matched pairs. PQ = RQ * SQ.
    """

    rq: float
    sq: float
    pq: float


Span = tuple[int, int]


def _validate(true: Sequence[bool], pred: Sequence[bool]) -> None:
    if len(true) != len(pred):
        raise ValueError(f"Length mismatch: true={len(true)}, pred={len(pred)}")
    if not true or not pred:
        raise ValueError("Empty boundary vector")
    if not true[0] or not pred[0]:
        raise ValueError("boundaries[0] must be True for both true and pred")


def boundaries_to_spans(boundaries: Sequence[bool]) -> list[Span]:
    """Convert a boundary vector to [start, end) spans."""
    starts = [i for i, b in enumerate(boundaries) if b]
    ends = [*starts[1:], len(boundaries)]
    return list(zip(starts, ends, strict=True))


def boundaries_to_doc_ids(boundaries: Sequence[bool]) -> list[int]:
    """Convert a boundary vector to per-page document IDs."""
    doc_id = -1
    out: list[int] = []
    for b in boundaries:
        if b:
            doc_id += 1
        out.append(doc_id)
    return out


def page_metrics(true: Sequence[bool], pred: Sequence[bool]) -> PRF:
    """Per-page boundary precision, recall, F1.

    By convention `boundaries[0]` is always True, so the first page is excluded
    from the boundary class metrics: it is trivially correct and would inflate
    scores. This matches the prevailing convention in the PSS literature.
    """
    _validate(true, pred)
    tp = sum(1 for t, p in zip(true[1:], pred[1:], strict=True) if t and p)
    fp = sum(1 for t, p in zip(true[1:], pred[1:], strict=True) if not t and p)
    fn = sum(1 for t, p in zip(true[1:], pred[1:], strict=True) if t and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1)


def _iou(a: Span, b: Span) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def panoptic_quality(true: Sequence[bool], pred: Sequence[bool]) -> PanopticQuality:
    """Document-level RQ, SQ, PQ with IoU greater than 0.5 matching.

    A predicted document matches a true document if their IoU exceeds 0.5;
    the threshold guarantees at most one match per side. RQ is the F1 over
    matched pairs, SQ is the mean IoU of matched pairs, PQ = RQ * SQ.
    """
    _validate(true, pred)
    true_spans = boundaries_to_spans(true)
    pred_spans = boundaries_to_spans(pred)

    matched_iou: list[float] = []
    matched_true: set[int] = set()
    matched_pred: set[int] = set()
    for ti, t_span in enumerate(true_spans):
        for pi, p_span in enumerate(pred_spans):
            if pi in matched_pred:
                continue
            iou = _iou(t_span, p_span)
            if iou > 0.5:
                matched_iou.append(iou)
                matched_true.add(ti)
                matched_pred.add(pi)
                break

    tp = len(matched_iou)
    fp = len(pred_spans) - tp
    fn = len(true_spans) - tp
    rq = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    sq = sum(matched_iou) / tp if tp else 0.0
    pq = rq * sq
    return PanopticQuality(rq=rq, sq=sq, pq=pq)


def stp(pairs: Sequence[tuple[Sequence[bool], Sequence[bool]]]) -> float:
    """Straight-through processing: fraction of streams predicted exactly."""
    if not pairs:
        raise ValueError("STP requires at least one (true, pred) pair")
    return sum(1 for t, p in pairs if tuple(t) == tuple(p)) / len(pairs)


def mndd(true: Sequence[bool], pred: Sequence[bool]) -> int:
    """Minimum number of page drag-and-drops to fix the prediction.

    From Mungmeeprued et al. 2022. Reflects user-visible repair effort with
    document ordering allowed to shuffle. We use a greedy assignment of
    predicted documents to true documents (largest predicted first, claim
    the highest-overlap unclaimed true doc) and count pages whose predicted
    document maps to a different true document than their actual one. The
    greedy heuristic matches common practice; a Hungarian-optimal version
    can replace this without changing the interface.
    """
    _validate(true, pred)
    true_ids = boundaries_to_doc_ids(true)
    pred_ids = boundaries_to_doc_ids(pred)

    pred_pages: dict[int, list[int]] = {}
    for i, pid in enumerate(pred_ids):
        pred_pages.setdefault(pid, []).append(i)

    assignment: dict[int, int] = {}
    claimed_true: set[int] = set()
    for pid in sorted(pred_pages, key=lambda p: -len(pred_pages[p])):
        counts: dict[int, int] = {}
        for page in pred_pages[pid]:
            counts[true_ids[page]] = counts.get(true_ids[page], 0) + 1
        for tid in sorted(counts, key=lambda t: -counts[t]):
            if tid not in claimed_true:
                assignment[pid] = tid
                claimed_true.add(tid)
                break

    return sum(1 for i in range(len(true)) if assignment.get(pred_ids[i]) != true_ids[i])
