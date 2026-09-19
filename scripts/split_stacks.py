"""Write each labeled stack as a corpus of its own.

Two real stacks is enough for a leave-one-out experiment and nothing more:
train with one of them alongside the composed corpus, measure on the other,
then swap. It answers whether a handful of genuinely scanned documents moves
the model at all, which decides whether collecting more is worth the work.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from papercut.data.loaders.hf import HfPssCorpus


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    corpus = HfPssCorpus.load_from_disk(args.corpus)
    for index, stream in enumerate(corpus.streams):
        pages = set(stream.pages)
        single = HfPssCorpus(
            streams=[stream],
            _texts={p: t for p, t in corpus._texts.items() if p in pages},
            _layouts={p: v for p, v in corpus._layouts.items() if p in pages},
            _visuals={p: v for p, v in corpus._visuals.items() if p in pages},
        )
        path = args.out_prefix.with_name(f"{args.out_prefix.name}-{index}.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        single.save(path)
        print(f"{path}: {len(stream.pages)} pages, {sum(stream.boundaries)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
