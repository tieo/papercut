"""Evaluate a saved model the way it is actually served, with uncertainty.

Serving wraps the raw splitter in the blank-page gate, so a number measured
on the bare model describes something nobody runs. This evaluates any
combination of the gate and Viterbi decoding, and reports a bootstrap
interval over documents rather than a bare mean: a corpus of 400 sampled
streams can be built from a few dozen documents, and resampling documents is
what says how far the number would move on another draw of the same mail.

Example:
    nix-shell --run '.venv/bin/python scripts/eval_model.py \\
        --model data/models/layout_sem_vis_personal_only.pkl \\
        --test data/private/personal-mail-test.pkl --gated --bootstrap 1000'
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from papercut.data.loaders.hf import HfPssCorpus
from papercut.eval.metrics import mndd, page_metrics, panoptic_quality, stp
from papercut.models.baselines.tfidf_xgb_layout_sem_vis import TfIdfXgbLayoutSemVis
from papercut.models.smoothing.blank_gate import BlankPageGated
from papercut.models.smoothing.viterbi import SequenceSmoothed
from papercut.streams.types import Stream


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train", type=Path, help="Corpus whose boundary statistics fit Viterbi.")
    parser.add_argument("--gated", action="store_true", help="Wrap in the blank-page gate.")
    parser.add_argument("--viterbi", action="store_true", help="Decode with the 2-state HMM.")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def _document_starts(stream: Stream) -> list[tuple[str, int]]:
    """The document each boundary opens, named by its source and page."""
    return [
        (page.source, index)
        for index, (page, flag) in enumerate(zip(stream.pages, stream.boundaries, strict=True))
        if flag
    ]


def _per_document_hits(streams: Sequence[Stream], predictions: Sequence[tuple[bool, ...]]):
    """Whether each document instance was opened exactly where it begins.

    A document counts as recovered when the boundary at its first page is
    predicted and no boundary is predicted inside it, which is the stream
    level "straight through" condition narrowed to one document.
    """
    hits: dict[str, list[bool]] = {}
    for stream, pred in zip(streams, predictions, strict=True):
        starts = [index for index, flag in enumerate(stream.boundaries) if flag]
        bounds = [*starts, len(stream.pages)]
        for position, start in enumerate(starts):
            end = bounds[position + 1]
            correct = pred[start] and not any(pred[start + 1 : end])
            hits.setdefault(stream.pages[start].source, []).append(bool(correct))
    return hits


def _bootstrap_interval(hits: dict[str, list[bool]], draws: int, seed: int) -> dict[str, float]:
    """Resample documents with replacement to bound the recovery rate."""
    sources = list(hits)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(draws):
        sample = [rng.choice(sources) for _ in sources]
        values = [value for source in sample for value in hits[source]]
        means.append(sum(values) / len(values))
    means.sort()
    return {
        "low": means[int(0.025 * len(means))],
        "high": means[min(len(means) - 1, int(0.975 * len(means)))],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    test = HfPssCorpus.load_from_disk(args.test)
    model: Any = TfIdfXgbLayoutSemVis.load_with_corpus(args.model, test)
    if args.threshold is not None:
        model.threshold = args.threshold
    label = ["raw"]

    if args.gated:
        model = BlankPageGated(submodel=model, corpus=test)
        label.append("gated")
    if args.viterbi:
        smoothed = SequenceSmoothed(submodel=model)
        if args.train is not None:
            smoothed.fit(HfPssCorpus.load_from_disk(args.train).streams)
        else:
            smoothed.fit(test.streams)
        model = smoothed
        label.append("viterbi")

    predictions = [model.predict_boundaries(stream) for stream in test.streams]
    pairs = [(stream.boundaries, pred) for stream, pred in zip(test.streams, predictions, strict=True)]
    hits = _per_document_hits(test.streams, predictions)
    document_rate = mean(value for values in hits.values() for value in values)

    metrics: dict[str, Any] = {
        "model": str(args.model),
        "test": str(args.test),
        "wrapping": "+".join(label),
        "threshold": args.threshold,
        "n_streams": len(test.streams),
        "n_documents": len(hits),
        "stp": stp(pairs),
        "page_f1_mean": mean(page_metrics(t, p).f1 for t, p in pairs),
        "pq_mean": mean(panoptic_quality(t, p).pq for t, p in pairs),
        "mndd_mean": mean(mndd(t, p) for t, p in pairs),
        "document_recovery": document_rate,
    }
    if args.bootstrap:
        metrics["document_recovery_ci"] = _bootstrap_interval(hits, args.bootstrap, args.seed)

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
