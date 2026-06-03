from __future__ import annotations

from pathlib import Path

import pytest

from papercut.cli.main import main
from papercut.data.loaders.hf import HfPssSchema, corpus_from_rows


def _make_corpus_pickle(path: Path) -> None:
    rich_rows = []
    for s in range(8):
        rich_rows.append(
            {
                "stream_id": f"S{s}",
                "page_ids": [f"S{s}/p{i}" for i in range(5)],
                "boundaries": [True, False, True, False, True],
                "texts": [
                    f"acme corp letterhead {s} chapter 1 intro overview",
                    f"acme corp body continues section {s} discussion",
                    f"recipe roast duck cherry sauce {s} ingredients",
                    f"recipe body ingredient continues simmer reduce {s}",
                    f"weather report cloudy {s} precipitation forecast",
                ],
            }
        )
    schema = HfPssSchema(
        boundaries_col="boundaries",
        stream_id_col="stream_id",
        page_ids_col="page_ids",
        text_col="texts",
    )
    corpus = corpus_from_rows(rich_rows, schema, "ns")
    corpus.save(path)


def test_eval_run_trivial(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "corpus.pkl"
    _make_corpus_pickle(p)
    rc = main(["eval", "run", "--corpus", str(p), "--model", "trivial:every-page"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "page_f1=" in captured.out


def test_eval_run_text_similarity(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "corpus.pkl"
    _make_corpus_pickle(p)
    rc = main(["eval", "run", "--corpus", str(p), "--model", "text-similarity"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "page_f1=" in captured.out


def test_eval_run_tfidf_xgb(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("xgboost")
    p = tmp_path / "corpus.pkl"
    _make_corpus_pickle(p)
    rc = main(["eval", "run", "--corpus", str(p), "--model", "tfidf-xgb"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Fitting tfidf-xgb" in captured.out
    assert "page_f1=" in captured.out


def test_eval_run_missing_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "eval",
            "run",
            "--corpus",
            str(tmp_path / "missing.pkl"),
            "--model",
            "trivial:every-page",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "not found" in captured.err
