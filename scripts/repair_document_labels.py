"""Split corpus documents that a printed page count shows to be several.

A corpus built from files takes one file as one document, which is an
assumption about how the files were made rather than an observation. Archive
scans break it: a single PDF holds an invoice and its attachment, or a run of
statements, and every boundary inside it is then labelled continuation. A
model fitted on that is taught to merge exactly the case it exists to split,
and scored for doing so.

Where a page announces "1 of N" after a page that announced the last of its
own count, the file states its own seam, and this splits the document there.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

from papercut.data.loaders.hf import HfPssCorpus
from papercut.models.baselines.tfidf_xgb_layout import _pagination
from papercut.streams.compose import documents_from_corpus
from papercut.streams.concat import poisson
from papercut.streams.types import PageRef, Stream


def seams(texts: Sequence[str]) -> list[int]:
    """Indices where one document's page count ends and another's begins.

    The evidence has to be a count that reached its own total followed by a
    count starting again, because a count that merely restarts is the ordinary
    shape of front matter giving way to a body: a manual numbers its contents
    in roman numerals and begins the text at one, inside a single document.
    Accepting a bare restart split six real documents when this was first
    written, every one of them at a front matter or continuation page.
    """
    marks = [_pagination(text) for text in texts]
    found = []
    for index in range(1, len(marks)):
        found_here, number, total = marks[index]
        before_found, before_number, before_total = marks[index - 1]
        restarts = found_here and number == 1 and total > 1
        closed = before_found and before_number == before_total and before_total > 1
        if restarts and closed:
            found.append(index)
    return found


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=6000)
    parser.add_argument("--mean-documents", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=71)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    corpus = HfPssCorpus.load_from_disk(args.corpus)

    repaired: list[tuple[str, tuple[PageRef, ...]]] = []
    split_count = 0
    for source, pages in documents_from_corpus(corpus):
        cuts = seams([corpus.text(page) for page in pages])
        if not cuts:
            repaired.append((source, pages))
            continue
        split_count += 1
        bounds = [0, *cuts, len(pages)]
        for part, (start, stop) in enumerate(pairwise(bounds)):
            repaired.append((f"{source}#part{part}", pages[start:stop]))

    print(f"{split_count} documents split into {len(repaired)} from {len(corpus.streams)} streams")

    rng = random.Random(args.seed)
    streams: list[Stream] = []
    for _ in range(args.streams):
        count = min(len(repaired), max(1, poisson(rng, args.mean_documents)))
        pages: list[PageRef] = []
        boundaries: list[bool] = []
        for _, document_pages in rng.sample(repaired, k=count):
            pages.extend(document_pages)
            boundaries.extend([True, *([False] * (len(document_pages) - 1))])
        streams.append(Stream(pages=tuple(pages), boundaries=tuple(boundaries)))

    out = HfPssCorpus(
        streams=streams,
        _texts=dict(corpus._texts),
        _layouts=dict(corpus._layouts),
        _visuals=dict(corpus._visuals),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"wrote {args.out}: {len(repaired)} documents, {len(streams)} streams")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
