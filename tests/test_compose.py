from __future__ import annotations

import random

from papercut.data.loaders.hf import HfPssCorpus
from papercut.streams.compose import compose_corpus, documents_from_corpus
from papercut.streams.types import PageRef, Stream


def _corpus() -> HfPssCorpus:
    documents = {"short": 2, "medium": 4, "archive": 30}
    pages = {name: tuple(PageRef(source=name, page=i) for i in range(n)) for name, n in documents.items()}
    streams = [
        Stream(
            pages=pages["short"] + pages["archive"],
            boundaries=(True, False, True, *([False] * 29)),
        ),
        Stream(pages=pages["medium"], boundaries=(True, False, False, False)),
    ]
    flat = [page for group in pages.values() for page in group]
    return HfPssCorpus(
        streams=streams,
        _texts={page: f"{page.source}-{page.page}" for page in flat},
        _layouts={page: [float(page.page)] for page in flat},
        _visuals={page: [0.5] for page in flat},
    )


def test_documents_recovered_once_in_page_order() -> None:
    documents = dict(documents_from_corpus(_corpus()))
    assert set(documents) == {"short", "medium", "archive"}
    assert [page.page for page in documents["medium"]] == [0, 1, 2, 3]
    assert len(documents["archive"]) == 30


def test_long_documents_are_dropped_and_features_follow() -> None:
    composed = compose_corpus(
        _corpus(),
        n_streams=20,
        mean_documents_per_stream=1.5,
        max_document_pages=10,
        seed=7,
    )
    sources = {page.source for stream in composed.streams for page in stream.pages}
    assert sources == {"short", "medium"}
    for stream in composed.streams:
        for page in stream.pages:
            assert composed.text(page)
            assert composed.layout(page)
            assert composed.visual(page)


def test_every_document_starts_at_a_boundary() -> None:
    composed = compose_corpus(
        _corpus(), n_streams=30, mean_documents_per_stream=2.0, max_document_pages=10, seed=1
    )
    for stream in composed.streams:
        starts = [i for i, flag in enumerate(stream.boundaries) if flag]
        assert starts[0] == 0
        for start in starts:
            assert stream.pages[start].page == 0


def test_composition_is_deterministic_for_a_seed() -> None:
    kwargs = {"n_streams": 10, "mean_documents_per_stream": 1.5, "max_document_pages": 10, "seed": 3}
    first = compose_corpus(_corpus(), **kwargs)
    second = compose_corpus(_corpus(), **kwargs)
    assert [s.pages for s in first.streams] == [s.pages for s in second.streams]
    assert random.Random(3).random() == random.Random(3).random()


def test_clusters_group_look_alike_first_pages() -> None:
    from papercut.streams.compose import cluster_documents

    pages = {
        name: tuple(PageRef(source=name, page=i) for i in range(2))
        for name in ("invoice-a", "invoice-b", "letter")
    }
    texts = {}
    for name, group in pages.items():
        opening = "Vodafone GmbH Rechnung Kundennummer" if name.startswith("invoice") else "Finanzamt Bescheid"
        texts[group[0]] = opening
        texts[group[1]] = "continuation"
    corpus = HfPssCorpus(streams=[Stream(pages=group, boundaries=(True, False)) for group in pages.values()], _texts=texts)
    clusters = cluster_documents(corpus, [(name, group) for name, group in pages.items()])
    sizes = sorted(len(group) for group in clusters)
    assert sizes == [1, 2]


def test_same_sender_streams_draw_from_one_cluster() -> None:
    from papercut.streams.compose import corpus_from_documents, documents_from_corpus

    pages = {
        name: tuple(PageRef(source=name, page=i) for i in range(2))
        for name in ("invoice-a", "invoice-b", "invoice-c", "letter")
    }
    texts = {}
    for name, group in pages.items():
        texts[group[0]] = "Vodafone Rechnung" if name.startswith("invoice") else "Finanzamt Bescheid"
        texts[group[1]] = "continuation"
    corpus = HfPssCorpus(
        streams=[Stream(pages=group, boundaries=(True, False)) for group in pages.values()],
        _texts=texts,
        _layouts={page: [0.0] for group in pages.values() for page in group},
        _visuals={page: [0.0] for group in pages.values() for page in group},
    )
    composed = corpus_from_documents(
        corpus,
        documents_from_corpus(corpus),
        n_streams=40,
        mean_documents_per_stream=2.0,
        seed=5,
        same_sender_probability=1.0,
    )
    for stream in composed.streams:
        sources = {page.source for page in stream.pages}
        assert all(source.startswith("invoice") for source in sources)
