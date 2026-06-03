from __future__ import annotations

from pathlib import Path

import pytest

from papercut.cli.main import main
from papercut.data.loaders.hf import HfPssSchema, corpus_from_rows


def _make_corpus(path: Path, n_streams: int) -> None:
    rows = []
    for s in range(n_streams):
        rows.append(
            {
                "stream_id": f"S{s}",
                "page_ids": [f"S{s}/p{i}" for i in range(5)],
                "boundaries": [True, False, True, False, True],
                "texts": [
                    f"acme corp letterhead {s} chapter intro",
                    f"acme corp body section {s} continues",
                    f"recipe roast duck {s} ingredients list",
                    f"recipe body simmer reduce {s}",
                    f"weather report cloudy {s} forecast",
                ],
            }
        )
    schema = HfPssSchema(
        boundaries_col="boundaries",
        stream_id_col="stream_id",
        page_ids_col="page_ids",
        text_col="texts",
    )
    corpus_from_rows(rows, schema, "ns").save(path)


def test_prospective_runs_with_trivial_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "corpus.pkl"
    _make_corpus(p, n_streams=12)
    rc = main(
        [
            "eval",
            "prospective",
            "--corpus",
            str(p),
            "--models",
            "trivial:every-page,trivial:never-split",
            "--slices",
            "3",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "every_page_new_doc" in captured.out
    assert "never_split" in captured.out
    assert "s1" in captured.out
    assert "s2" in captured.out


def test_prospective_with_trainable_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("xgboost")
    p = tmp_path / "corpus.pkl"
    _make_corpus(p, n_streams=16)
    rc = main(
        [
            "eval",
            "prospective",
            "--corpus",
            str(p),
            "--models",
            "tfidf-xgb",
            "--slices",
            "4",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "tfidf_xgb" in captured.out


def test_prospective_rejects_too_few_streams(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "corpus.pkl"
    _make_corpus(p, n_streams=3)
    rc = main(
        [
            "eval",
            "prospective",
            "--corpus",
            str(p),
            "--models",
            "trivial:never-split",
            "--slices",
            "5",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "streams" in captured.err
