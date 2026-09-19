"""Build a corpus of real scanner stacks in their true page order.

A sampled corpus composes streams out of separate documents, which is a
stand-in for a scan. A labeled stack is the real thing: the pages a feeder
produced in one pass, in the order they came out, with the document starts
written down by hand. This turns those stacks into a corpus so a model can be
measured on the input it will actually receive.

The labels file maps each stack's path to the 1-based page numbers that begin
a document, the same format `build_personal_corpus.py` consumes.

Example:
    nix-shell --run '.venv/bin/python scripts/build_stack_corpus.py \\
        --labels data/private/ground_truth.json \\
        --out data/private/real-stacks.pkl'
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from papercut.data.loaders.hf import HfPssCorpus
from papercut.serve.pdf_input import pdf_input
from papercut.streams.types import PageRef, Stream


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--languages", default="deu+eng")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--pdftoppm")
    parser.add_argument("--tesseract")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    labels = json.loads(args.labels.read_text())["stacks"]

    streams: list[Stream] = []
    texts: dict[PageRef, str] = {}
    layouts: dict[PageRef, list[float]] = {}
    visuals: dict[PageRef, list[float]] = {}

    for path_text, starts in sorted(labels.items()):
        path = Path(path_text)
        identifier = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        parsed = pdf_input(
            path,
            languages=args.languages,
            dpi=args.dpi,
            pdftoppm_path=args.pdftoppm,
            tesseract_path=args.tesseract,
        )
        source_pages = parsed.stream.pages
        starts_zero = {number - 1 for number in starts}
        if max(starts_zero) >= len(source_pages):
            raise ValueError(f"{path.name}: a document start is past the last page")

        pages = tuple(PageRef(source=f"stack/{identifier}", page=i) for i in range(len(source_pages)))
        for page, original in zip(pages, source_pages, strict=True):
            texts[page] = parsed.corpus.text(original)
            layouts[page] = parsed.corpus.layout(original)
            visuals[page] = parsed.corpus.visual(original)
        streams.append(
            Stream(
                pages=pages,
                boundaries=tuple(i in starts_zero for i in range(len(pages))),
            )
        )
        print(
            f"stack {identifier}: {len(pages)} pages, {len(starts_zero)} documents",
            flush=True,
        )

    corpus = HfPssCorpus(streams=streams, _texts=texts, _layouts=layouts, _visuals=visuals)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    corpus.save(args.out)
    print(f"wrote {args.out} with {len(streams)} stacks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
