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


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    cleaned = " ".join(text.lower().split())
    return {cleaned[i : i + n] for i in range(max(0, len(cleaned) - n + 1))}


def cluster_documents(
    corpus: HfPssCorpus,
    documents: Sequence[Document],
    threshold: float = 0.25,
    head: int = 600,
) -> list[list[Document]]:
    """Group documents whose first page looks like the same sender.

    A scanner stack is what arrived in one batch, so it carries runs of
    near-identical documents: three invoices from one provider, a statement
    series, a form sent twice. Sampling documents uniformly never builds that
    case, and it is the case where letterhead similarity argues for "same
    document" exactly when a new one starts. Grouping on the opening of the
    first page is enough to recreate those runs.
    """
    signatures = [
        (document, _char_ngrams(corpus.text(pages[0])[:head]))
        for document, pages in ((doc, doc[1]) for doc in documents)
    ]
    clusters: list[list[Document]] = []
    centroids: list[set[str]] = []
    for document, signature in signatures:
        placed = False
        for index, centroid in enumerate(centroids):
            union = len(signature | centroid)
            score = len(signature & centroid) / union if union else 0.0
            if score >= threshold:
                clusters[index].append(document)
                placed = True
                break
        if not placed:
            clusters.append([document])
            centroids.append(signature)
    return clusters


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


def partition_documents(
    documents: Sequence[Document], fraction: float, seed: int = 0
) -> tuple[list[Document], list[Document]]:
    """Split documents in two disjoint groups, the first holding `fraction`.

    Tuning a threshold on the same documents a number is reported against
    turns a holdout into a training set. Splitting by document, not by
    stream, keeps a document's pages out of both sides at once.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    ordered = sorted(documents, key=lambda item: item[0])
    random.Random(seed).shuffle(ordered)
    cut = max(1, round(len(ordered) * fraction))
    if cut >= len(ordered):
        raise ValueError("fraction leaves nothing on the other side")
    return ordered[:cut], ordered[cut:]


def corpus_from_documents(
    corpus: HfPssCorpus,
    documents: Sequence[Document],
    *,
    n_streams: int,
    mean_documents_per_stream: float,
    seed: int = 0,
    same_sender_probability: float = 0.0,
    cluster_threshold: float = 0.25,
) -> HfPssCorpus:
    """Sample streams over a chosen set of the corpus documents.

    `same_sender_probability` draws that share of streams from one cluster of
    look-alike documents, which is what a batch of mail from one provider
    looks like coming off the feeder.
    """
    if not documents:
        raise ValueError("No documents to compose streams from")
    rng = random.Random(seed)
    clusters = (
        [group for group in cluster_documents(corpus, documents, cluster_threshold) if len(group) > 1]
        if same_sender_probability > 0
        else []
    )
    streams: list[Stream] = []
    for _ in range(n_streams):
        pool = list(documents)
        if clusters and rng.random() < same_sender_probability:
            pool = list(rng.choice(clusters))
        count = min(len(pool), max(1, poisson(rng, mean_documents_per_stream)))
        pages: list[PageRef] = []
        boundaries: list[bool] = []
        for _, document_pages in rng.sample(pool, k=count):
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


def compose_corpus(
    corpus: HfPssCorpus,
    *,
    n_streams: int,
    mean_documents_per_stream: float,
    max_document_pages: int | None = None,
    seed: int = 0,
    same_sender_probability: float = 0.0,
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

    return corpus_from_documents(
        corpus,
        documents,
        n_streams=n_streams,
        mean_documents_per_stream=mean_documents_per_stream,
        seed=seed,
        same_sender_probability=same_sender_probability,
    )


def document_page_counts(documents: Sequence[Document]) -> list[int]:
    """Page count per document, for reporting a corpus composition."""
    return [len(pages) for _, pages in documents]
