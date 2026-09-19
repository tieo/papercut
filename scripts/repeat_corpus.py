"""Repeat a corpus's streams so a small set can carry weight in training.

Boosting is the cheap test of whether scarce data is useful or merely scarce.
Seven real scanned documents inside a hundred composed ones are 5 percent of
the training mass and change nothing; the same seven repeated until they are a
third of it either move the model or prove that collecting more will not.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from papercut.data.loaders.hf import HfPssCorpus


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--times", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.times < 1:
        raise ValueError("--times must be at least one")
    corpus = HfPssCorpus.load_from_disk(args.corpus)
    repeated = HfPssCorpus(
        streams=list(corpus.streams) * args.times,
        _texts=dict(corpus._texts),
        _layouts=dict(corpus._layouts),
        _visuals=dict(corpus._visuals),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    repeated.save(args.out)
    print(f"{args.out}: {len(repeated.streams)} streams from {len(corpus.streams)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
