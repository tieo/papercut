"""Recompose a saved private corpus into scanner shaped streams.

The corpus built from local PDFs mixes short mail with archive scans hundreds
of pages long, and its streams drew up to twelve documents each. A real
scanner stack carries a handful of short documents, so metrics measured on the
original composition are dominated by documents that never reach the scanner.

This rebuilds the streams from the documents already stored in the pickles, so
no OCR runs again, and keeps the existing disjoint split between the training
and holdout documents.

Example:
    nix-shell --run '.venv/bin/python scripts/recompose_personal_corpus.py \\
        --corpus data/private/personal-train.pkl \\
        --out data/private/personal-mail-train.pkl \\
        --streams 1500 --mean-documents 1.5 --max-document-pages 20'
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from papercut.data.loaders.hf import HfPssCorpus
from papercut.streams.compose import compose_corpus, document_page_counts, documents_from_corpus


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=1500)
    parser.add_argument("--mean-documents", type=float, default=1.5)
    parser.add_argument("--max-document-pages", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _describe(name: str, documents: Sequence[tuple[str, tuple]]) -> None:
    counts = sorted(document_page_counts(documents))
    if not counts:
        print(f"{name}: no documents")
        return
    total = sum(counts)
    print(
        f"{name}: {len(counts)} documents, {total} pages, "
        f"median {counts[len(counts) // 2]}, max {counts[-1]}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    corpus = HfPssCorpus.load_from_disk(args.corpus)
    documents = documents_from_corpus(corpus)
    _describe("source", documents)

    composed = compose_corpus(
        corpus,
        n_streams=args.streams,
        mean_documents_per_stream=args.mean_documents,
        max_document_pages=args.max_document_pages,
        seed=args.seed,
    )
    _describe("kept", documents_from_corpus(composed))

    pages = [len(stream.pages) for stream in composed.streams]
    boundaries = sum(sum(stream.boundaries) for stream in composed.streams)
    print(
        f"streams: {len(composed.streams)}, pages {sum(pages)}, "
        f"pages per stream median {sorted(pages)[len(pages) // 2]}, "
        f"boundary rate {boundaries / sum(pages):.3f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    composed.save(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
