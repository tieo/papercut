"""Evaluate a saved semantic and visual XGBoost model on a corpus.

Example:
    nix-shell --run '.venv/bin/python scripts/eval_saved_sem_vis.py \\
        --model data/models/layout_sem_vis_maxgiga.pkl \\
        --test data/maxgiga_test.pkl'
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from papercut.data.loaders.hf import HfPssCorpus
from papercut.eval.runner import evaluate
from papercut.models.baselines.tfidf_xgb_layout_sem_vis import TfIdfXgbLayoutSemVis


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    test = HfPssCorpus.load_from_disk(args.test)
    model = TfIdfXgbLayoutSemVis.load_with_corpus(args.model, test)
    report = evaluate(model, test.streams)
    metrics = report._asdict()
    metrics.update({"model": str(args.model), "test": str(args.test)})
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        print(f"Saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
