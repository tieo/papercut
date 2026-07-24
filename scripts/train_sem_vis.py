"""Train and evaluate the semantic and visual XGBoost page splitter.

The command works from saved ``HfPssCorpus`` files. It keeps the training
configuration alongside the code and writes the fitted model only after its
holdout metrics have been reported.

Example:
    nix-shell --run '.venv/bin/python scripts/train_sem_vis.py \\
        --train data/maxgiga_train.pkl --test data/maxgiga_test.pkl \\
        --out data/models/layout_sem_vis_charngram_maxgiga.pkl \\
        --analyzer char_wb --ngram-range 3 5 --max-features 20000 \\
        --n-estimators 500 --learning-rate 0.04 --colsample-bytree 0.5'
"""

from __future__ import annotations

import argparse
import gc
from collections.abc import Sequence
from pathlib import Path

from papercut.data.loaders.hf import HfPssCorpus
from papercut.eval.runner import evaluate
from papercut.models.baselines.tfidf_xgb_layout_sem_vis import TfIdfXgbLayoutSemVis


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Training corpus pickle.")
    parser.add_argument("--test", type=Path, required=True, help="Holdout corpus pickle.")
    parser.add_argument("--out", type=Path, required=True, help="Destination model pickle.")
    parser.add_argument("--analyzer", choices=("word", "char", "char_wb"), default="word")
    parser.add_argument("--ngram-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=(1, 2))
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--colsample-bytree", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def _combined_corpus(train: HfPssCorpus, test: HfPssCorpus) -> HfPssCorpus:
    return HfPssCorpus(
        streams=[*train.streams, *test.streams],
        _texts={**train._texts, **test._texts},
        _layouts={**train._layouts, **test._layouts},
        _visuals={**train._visuals, **test._visuals},
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 0 < args.colsample_bytree <= 1:
        raise ValueError("--colsample-bytree must be in (0, 1]")

    print("Loading corpora", flush=True)
    train = HfPssCorpus.load_from_disk(args.train)
    test = HfPssCorpus.load_from_disk(args.test)
    corpus = _combined_corpus(train, test)
    del train
    gc.collect()

    print(
        "Fitting "
        f"streams={len(corpus.streams) - len(test.streams)} "
        f"analyzer={args.analyzer} ngrams={tuple(args.ngram_range)} "
        f"features={args.max_features} trees={args.n_estimators} "
        f"depth={args.max_depth} colsample={args.colsample_bytree}",
        flush=True,
    )
    model = TfIdfXgbLayoutSemVis(
        corpus=corpus,
        analyzer=args.analyzer,
        ngram_range=tuple(args.ngram_range),
        max_features=args.max_features,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        colsample_bytree=args.colsample_bytree,
        threshold=args.threshold,
    )
    model.fit(corpus.streams[: -len(test.streams)])
    print("Evaluating holdout", flush=True)
    report = evaluate(model, test.streams)
    print(
        f"STP={report.stp:.4f} page_f1={report.page_f1_mean:.4f} "
        f"pq={report.pq_mean:.4f} mndd={report.mndd_mean:.4f}",
        flush=True,
    )
    model.save(args.out)
    print(f"Saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
