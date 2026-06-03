from __future__ import annotations

from pathlib import Path

from papercut.data.loaders.hf import HfPssCorpus, HfPssSchema, corpus_from_rows
from papercut.streams.types import PageRef


def _rows() -> list[dict[str, object]]:
    return [
        {
            "stream_id": "S0",
            "page_ids": ["S0/p0", "S0/p1"],
            "boundaries": [True, False],
            "texts": ["alpha header", "alpha body"],
        },
        {
            "stream_id": "S1",
            "page_ids": ["S1/p0", "S1/p1", "S1/p2"],
            "boundaries": [True, False, True],
            "texts": ["beta header", "beta body", "gamma header"],
        },
    ]


def test_save_and_load_round_trips(tmp_path: Path) -> None:
    schema = HfPssSchema(
        boundaries_col="boundaries",
        stream_id_col="stream_id",
        page_ids_col="page_ids",
        text_col="texts",
    )
    corpus = corpus_from_rows(_rows(), schema, "ns")
    path = tmp_path / "corpus.pkl"
    corpus.save(path)

    restored = HfPssCorpus.load_from_disk(path)
    assert len(restored.streams) == len(corpus.streams)
    assert restored.streams[0].boundaries == corpus.streams[0].boundaries
    assert restored.streams[0].pages == corpus.streams[0].pages

    page = corpus.streams[1].pages[2]
    assert restored.text(page) == corpus.text(page)


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    schema = HfPssSchema(boundaries_col="boundaries", stream_id_col="stream_id")
    corpus = corpus_from_rows(
        [{"stream_id": "X", "boundaries": [True, False]}],
        schema,
        "ns",
    )
    nested = tmp_path / "a" / "b" / "c" / "corpus.pkl"
    corpus.save(nested)
    assert nested.exists()
    restored = HfPssCorpus.load_from_disk(nested)
    assert restored.streams[0].pages[0] == PageRef("ns:X/0", 0)
