"""Rebuild stream corpora from the documents an existing corpus already holds.

A corpus pickle keeps its pages keyed by `PageRef.source`, so the documents
that went into it can be recovered without touching the original PDFs or
running OCR again. That makes the composition of a corpus (how many documents
land in one stream, and which documents are eligible at all) a parameter that
can be changed after the expensive step is done.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from papercut.data.loaders.hf import HfPssCorpus
from papercut.streams.concat import poisson
from papercut.streams.types import PageRef, Stream

Document = tuple[str, tuple[PageRef, ...]]


def documents_from_corpus(corpus: HfPssCorpus) -> list[Document]:
    """Recover the distinct documents behind a corpus, pages in reading order.

    A document is one `PageRef.source`; its pages are ordered by page index, so
    a document that appears in several streams is recovered once.
    """
    by_source: dict[str, set[PageRef]] = {}
    for stream in corpus.streams:
        for page in stream.pages:
            by_source.setdefault(page.source, set()).add(page)
    return [
        (source, tuple(sorted(pages, key=lambda page: page.page)))
        for source, pages in sorted(by_source.items())
    ]


def compose_corpus(
    corpus: HfPssCorpus,
    *,
    n_streams: int,
    mean_documents_per_stream: float,
    max_document_pages: int | None = None,
    seed: int = 0,
) -> HfPssCorpus:
    """Resample streams over the corpus documents, keeping page features.

    Each stream draws Poisson(`mean_documents_per_stream`) documents, at least
    one, without replacement inside the stream. `max_document_pages` drops
    documents longer than a scanner stack ever carries, which otherwise
    dominate both the page count and the boundary rate.
    """
    documents = documents_from_corpus(corpus)
    if max_document_pages is not None:
        documents = [doc for doc in documents if len(doc[1]) <= max_document_pages]
    if not documents:
        raise ValueError("No documents left to compose streams from")

    rng = random.Random(seed)
    streams: list[Stream] = []
    for _ in range(n_streams):
        count = min(len(documents), max(1, poisson(rng, mean_documents_per_stream)))
        pages: list[PageRef] = []
        boundaries: list[bool] = []
        for _, document_pages in rng.sample(documents, k=count):
            pages.extend(document_pages)
            boundaries.extend([True, *([False] * (len(document_pages) - 1))])
        streams.append(Stream(pages=tuple(pages), boundaries=tuple(boundaries)))

    kept = {page for _, document_pages in documents for page in document_pages}
    return HfPssCorpus(
        streams=streams,
        _texts={page: text for page, text in corpus._texts.items() if page in kept},
        _layouts={page: value for page, value in corpus._layouts.items() if page in kept},
        _visuals={page: value for page, value in corpus._visuals.items() if page in kept},
    )


def document_page_counts(documents: Sequence[Document]) -> list[int]:
    """Page count per document, for reporting a corpus composition."""
    return [len(pages) for _, pages in documents]
