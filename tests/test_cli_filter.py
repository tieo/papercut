from __future__ import annotations

from pathlib import Path

import pytest

from papercut.cli.main import main
from papercut.data.loaders.hf import HfPssCorpus, HfPssSchema, corpus_from_rows


def _build_mixed_corpus(path: Path) -> None:
    rows = []
    for s, length in enumerate([2, 5, 8, 12, 20, 3]):
        rows.append(
            {
                "stream_id": f"S{s}",
                "page_ids": [f"S{s}/p{i}" for i in range(length)],
                "boundaries": [True] + [False] * (length - 1),
                "texts": [f"text {s} {i}" for i in range(length)],
            }
        )
    schema = HfPssSchema(
        boundaries_col="boundaries",
        stream_id_col="stream_id",
        page_ids_col="page_ids",
        text_col="texts",
    )
    corpus_from_rows(rows, schema, "ns").save(path)


def test_filter_drops_long_streams(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "in.pkl"
    out = tmp_path / "out.pkl"
    _build_mixed_corpus(src)

    rc = main(["data", "filter", "--in", str(src), "--out", str(out), "--max-pages", "8"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Kept 4 / 6" in captured.out

    filtered = HfPssCorpus.load_from_disk(out)
    assert all(len(s) <= 8 for s in filtered.streams)
    assert all(len(s) >= 1 for s in filtered.streams)


def test_filter_min_pages(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "in.pkl"
    out = tmp_path / "out.pkl"
    _build_mixed_corpus(src)
    rc = main(
        [
            "data",
            "filter",
            "--in",
            str(src),
            "--out",
            str(out),
            "--min-pages",
            "5",
            "--max-pages",
            "12",
        ]
    )
    assert rc == 0
    filtered = HfPssCorpus.load_from_disk(out)
    assert all(5 <= len(s) <= 12 for s in filtered.streams)


def test_filter_missing_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        ["data", "filter", "--in", str(tmp_path / "missing.pkl"), "--out", str(tmp_path / "o")]
    )
    assert rc == 2


def test_filter_no_matches(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "in.pkl"
    out = tmp_path / "out.pkl"
    _build_mixed_corpus(src)
    rc = main(
        [
            "data",
            "filter",
            "--in",
            str(src),
            "--out",
            str(out),
            "--max-pages",
            "1",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "No streams matched" in captured.err
