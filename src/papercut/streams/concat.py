from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from papercut.streams.types import PageRef, Stream

SourceSpec = tuple[str, Path]


def concat_pdfs(sources: Sequence[SourceSpec], output_path: Path) -> Stream:
    """Concatenate PDFs into one stream, return the corresponding Stream.

    `sources` is a sequence of (stable_source_id, pdf_path) pairs in the order
    they should appear in the stream. Each PDF contributes its pages as one
    document, with a boundary at the first page of each source.
    """
    if not sources:
        raise ValueError("Cannot concatenate an empty source list")

    writer = PdfWriter()
    pages: list[PageRef] = []
    boundaries: list[bool] = []

    for source_id, path in sources:
        reader = PdfReader(str(path))
        n = len(reader.pages)
        if n == 0:
            raise ValueError(f"Source {source_id!r} at {path} has zero pages")
        for i, page in enumerate(reader.pages):
            writer.add_page(page)
            pages.append(PageRef(source=source_id, page=i))
            boundaries.append(i == 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        writer.write(f)

    return Stream(pages=tuple(pages), boundaries=tuple(boundaries))


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's algorithm. Returns k in [0, infinity)."""
    if lam <= 0:
        raise ValueError("lam must be positive")
    target = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p < target:
            return k - 1


def sample_stream_specs(
    source_pool: Sequence[SourceSpec],
    n_streams: int,
    mean_docs_per_stream: float,
    rng: random.Random,
) -> list[list[SourceSpec]]:
    """Sample stream compositions without doing any IO.

    Each stream draws Poisson(mean_docs_per_stream) source documents (lower-
    bounded by 1, upper-bounded by the pool size). Sampling is without
    replacement within a single stream so we never produce a "two copies of
    the same source" stream that would confuse boundary labels.
    """
    if n_streams <= 0:
        raise ValueError("n_streams must be positive")
    if not source_pool:
        raise ValueError("source_pool is empty")
    specs: list[list[SourceSpec]] = []
    for _ in range(n_streams):
        n_docs = max(1, _poisson(rng, mean_docs_per_stream))
        n_docs = min(n_docs, len(source_pool))
        specs.append(rng.sample(list(source_pool), k=n_docs))
    return specs


def build_stream_corpus(
    source_pool: Sequence[SourceSpec],
    output_dir: Path,
    n_streams: int = 100,
    mean_docs_per_stream: float = 10.0,
    seed: int = 0,
) -> list[Stream]:
    """Sample stream compositions and write each as a merged PDF in output_dir.

    Returns the list of labeled `Stream` objects aligned to the written PDFs
    (`output_dir/stream_<i>.pdf`).
    """
    rng = random.Random(seed)
    specs = sample_stream_specs(source_pool, n_streams, mean_docs_per_stream, rng)
    output_dir.mkdir(parents=True, exist_ok=True)
    streams: list[Stream] = []
    for i, spec in enumerate(specs):
        out_path = output_dir / f"stream_{i:06d}.pdf"
        streams.append(concat_pdfs(spec, out_path))
    return streams
