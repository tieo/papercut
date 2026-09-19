"""Report holdout metrics of a saved model across boundary thresholds.

The decision threshold trades split errors against merge errors, and the best
value depends on how many documents a stream carries. Page probabilities are
computed once and reused for every threshold, so a sweep costs one pass over
the corpus.

Example:
    nix-shell --run '.venv/bin/python scripts/sweep_threshold.py \\
        --model data/models/layout_sem_vis_maxgiga_plus_personal.pkl \\
        --test data/private/personal-mail-test.pkl'
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean

from papercut.data.loaders.hf import HfPssCorpus
from papercut.eval.metrics import mndd, page_metrics, panoptic_quality, stp
from papercut.models.baselines.tfidf_xgb_layout_sem_vis import TfIdfXgbLayoutSemVis


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    thresholds = args.thresholds or [round(0.05 * step, 2) for step in range(2, 19)]

    test = HfPssCorpus.load_from_disk(args.test)
    model = TfIdfXgbLayoutSemVis.load_with_corpus(args.model, test)
    probabilities = [model.predict_probs(stream) for stream in test.streams]

    rows = []
    for threshold in thresholds:
        pairs = []
        page_f1s = []
        pqs = []
        mndds = []
        for stream, probs in zip(test.streams, probabilities, strict=True):
            pred = tuple(i == 0 or prob >= threshold for i, prob in enumerate(probs))
            pairs.append((stream.boundaries, pred))
            page_f1s.append(page_metrics(stream.boundaries, pred).f1)
            pqs.append(panoptic_quality(stream.boundaries, pred).pq)
            mndds.append(mndd(stream.boundaries, pred))
        row = {
            "threshold": threshold,
            "stp": stp(pairs),
            "page_f1_mean": mean(page_f1s),
            "pq_mean": mean(pqs),
            "mndd_mean": mean(mndds),
        }
        rows.append(row)
        print(
            f"threshold={threshold:.2f} STP={row['stp']:.4f} page_f1={row['page_f1_mean']:.4f} "
            f"pq={row['pq_mean']:.4f} mndd={row['mndd_mean']:.4f}",
            flush=True,
        )

    best = max(rows, key=lambda row: row["stp"])
    print(f"best threshold {best['threshold']:.2f} at STP {best['stp']:.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {"model": str(args.model), "test": str(args.test), "rows": rows, "best": best},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
