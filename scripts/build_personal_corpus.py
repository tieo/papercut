"""Build a private document-disjoint PSS corpus from local PDF documents.

Every PDF supplied through ``--source-dir`` is one known document. ``--stack``
adds a scanned page stream whose true document starts are declared in a local
JSON file. The resulting pickles contain OCR, layout, visual features, and
anonymous page IDs only. Source paths and document text remain outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from papercut.data.loaders.hf import HfPssCorpus
from papercut.serve.pdf_input import pdf_input
from papercut.streams.concat import poisson
from papercut.streams.types import PageRef, Stream


@dataclass(frozen=True)
class PageFeatures:
    text: str
    layout: list[float]
    visual: list[float]


@dataclass(frozen=True)
class Document:
    identifier: str
    pages: tuple[PageFeatures, ...]


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_document(
    path: Path,
    identifier: str,
    *,
    languages: str,
    dpi: int,
    pdftoppm: str | None,
    tesseract: str | None,
) -> Document:
    parsed = pdf_input(
        path,
        languages=languages,
        dpi=dpi,
        pdftoppm_path=pdftoppm,
        tesseract_path=tesseract,
    )
    return Document(
        identifier=identifier,
        pages=tuple(
            PageFeatures(
                text=parsed.corpus.text(page),
                layout=parsed.corpus.layout(page),
                visual=parsed.corpus.visual(page),
            )
            for page in parsed.stream.pages
        ),
    )


def _split_stack(document: Document, starts: Sequence[int]) -> list[Document]:
    starts = tuple(starts)
    if not starts or starts[0] != 1:
        raise ValueError(f"Stack {document.identifier} must start at page 1")
    if any(a >= b for a, b in pairwise(starts)):
        raise ValueError(f"Stack {document.identifier} starts must be strictly increasing")
    if starts[-1] > len(document.pages):
        raise ValueError(f"Stack {document.identifier} start exceeds its page count")
    ends = (*starts[1:], len(document.pages) + 1)
    return [
        Document(
            identifier=f"{document.identifier}-{number:03d}",
            pages=document.pages[start - 1 : end - 1],
        )
        for number, (start, end) in enumerate(zip(starts, ends, strict=True), start=1)
    ]


def _stream_corpus(
    documents: Sequence[Document],
    n_streams: int,
    seed: int,
    mean_documents: float,
    max_document_pages: int | None,
) -> HfPssCorpus:
    """Compose streams the way a scanner stack arrives.

    Documents longer than `max_document_pages` are left out: an archive scan of
    several hundred pages carries most of the pages in the corpus while
    contributing one boundary, which buries the mail the splitter runs on.
    """
    if max_document_pages is not None:
        documents = [document for document in documents if len(document.pages) <= max_document_pages]
    if not documents:
        raise ValueError("Need at least one document")
    rng = random.Random(seed)
    features: dict[PageRef, PageFeatures] = {}
    document_pages: dict[str, tuple[PageRef, ...]] = {}
    for document in documents:
        pages = tuple(PageRef(source=f"personal/{document.identifier}", page=i) for i in range(len(document.pages)))
        document_pages[document.identifier] = pages
        features.update(dict(zip(pages, document.pages, strict=True)))

    streams: list[Stream] = []
    for _ in range(n_streams):
        count = min(len(documents), max(1, poisson(rng, mean_documents)))
        selected = rng.sample(list(documents), count)
        pages: list[PageRef] = []
        boundaries: list[bool] = []
        for document in selected:
            source_pages = document_pages[document.identifier]
            pages.extend(source_pages)
            boundaries.extend([True, *([False] * (len(source_pages) - 1))])
        streams.append(Stream(pages=tuple(pages), boundaries=tuple(boundaries)))

    return HfPssCorpus(
        streams=streams,
        _texts={page: item.text for page, item in features.items()},
        _layouts={page: item.layout for page, item in features.items()},
        _visuals={page: item.visual for page, item in features.items()},
    )


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--stack", type=Path, action="append", default=[])
    parser.add_argument("--exclude-dir", type=Path, action="append", default=[])
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--test-out", type=Path, required=True)
    parser.add_argument("--train-streams", type=int, default=500)
    parser.add_argument("--test-streams", type=int, default=150)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--mean-documents", type=float, default=1.5)
    parser.add_argument("--max-document-pages", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--languages", default="deu+eng")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pdftoppm")
    parser.add_argument("--tesseract")
    return parser.parse_args(argv)


def _pdfs(roots: Iterable[Path], excluded: set[Path], excluded_dirs: set[Path]) -> list[Path]:
    seen_hashes: set[str] = set()
    result: list[Path] = []
    for root in roots:
        for path in sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"
        ):
            if path.resolve() in excluded:
                continue
            if any(directory in path.resolve().parents for directory in excluded_dirs):
                continue
            digest = _digest(path)
            if digest not in seen_hashes:
                seen_hashes.add(digest)
                result.append(path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be between 0 and 1")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.labels is None:
        stack_labels: dict[Path, Sequence[int]] = {}
    else:
        labels = json.loads(args.labels.read_text())
        stack_labels = {Path(path).resolve(): starts for path, starts in labels["stacks"].items()}
    stack_paths = {path.resolve() for path in args.stack}
    if stack_paths != set(stack_labels):
        raise ValueError("--stack paths and labels.stacks keys must match")

    documents: list[Document] = []
    excluded_dirs = {path.resolve() for path in args.exclude_dir}
    skipped = 0
    source_paths = _pdfs(args.source_dir, excluded=stack_paths, excluded_dirs=excluded_dirs)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _parse_document,
                path,
                _digest(path)[:16],
                languages=args.languages,
                dpi=args.dpi,
                pdftoppm=args.pdftoppm,
                tesseract=args.tesseract,
            ): path
            for path in source_paths
        }
        for number, future in enumerate(as_completed(futures), start=1):
            try:
                documents.append(future.result())
                print("OCR document", number, flush=True)
            except (OSError, subprocess.CalledProcessError):
                skipped += 1
                print("Skipped unreadable PDF", file=sys.stderr, flush=True)
    for path in sorted(stack_paths):
        print("OCR labeled stack", flush=True)
        stack = _parse_document(
            path,
            _digest(path)[:16],
            languages=args.languages,
            dpi=args.dpi,
            pdftoppm=args.pdftoppm,
            tesseract=args.tesseract,
        )
        documents.extend(_split_stack(stack, stack_labels[path]))

    rng = random.Random(args.seed)
    rng.shuffle(documents)
    test_count = max(1, round(len(documents) * args.test_fraction))
    test_documents = documents[:test_count]
    train_documents = documents[test_count:]
    if len(train_documents) < 2 or len(test_documents) < 2:
        raise ValueError("Need at least two train and test documents")
    _stream_corpus(
        train_documents,
        args.train_streams,
        args.seed,
        args.mean_documents,
        args.max_document_pages,
    ).save(args.train_out)
    _stream_corpus(
        test_documents,
        args.test_streams,
        args.seed + 1,
        args.mean_documents,
        args.max_document_pages,
    ).save(args.test_out)
    print(
        f"Saved {len(train_documents)} train and {len(test_documents)} test documents "
        f"to {args.train_out} and {args.test_out}; skipped {skipped} unreadable PDFs",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
