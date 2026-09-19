"""Split a corpus into a tuning half and a reporting half, by document.

A threshold picked on the same documents the number is reported against is
not a holdout result. This divides the documents in two disjoint groups and
samples streams over each, so tuning happens on one set of mail and the
figure quoted comes from mail the choice never saw.

Example:
    nix-shell --run '.venv/bin/python scripts/split_holdout.py \\
        --corpus data/private/personal-mail-test.pkl \\
        --dev-out data/private/personal-mail-dev.pkl \\
        --test-out data/private/personal-mail-final.pkl'
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from papercut.data.loaders.hf import HfPssCorpus
from papercut.streams.compose import (
    corpus_from_documents,
    documents_from_corpus,
    partition_documents,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--dev-out", type=Path, required=True)
    parser.add_argument("--test-out", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.5)
    parser.add_argument("--streams", type=int, default=400)
    parser.add_argument("--mean-documents", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    corpus = HfPssCorpus.load_from_disk(args.corpus)
    dev_documents, test_documents = partition_documents(
        documents_from_corpus(corpus), args.dev_fraction, args.seed
    )
    for documents, path, seed in (
        (dev_documents, args.dev_out, args.seed),
        (test_documents, args.test_out, args.seed + 1),
    ):
        composed = corpus_from_documents(
            corpus,
            documents,
            n_streams=args.streams,
            mean_documents_per_stream=args.mean_documents,
            seed=seed,
        )
        pages = sum(len(stream.pages) for stream in composed.streams)
        path.parent.mkdir(parents=True, exist_ok=True)
        composed.save(path)
        print(f"{path}: {len(documents)} documents, {len(composed.streams)} streams, {pages} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
