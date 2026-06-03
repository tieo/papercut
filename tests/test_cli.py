from __future__ import annotations

from pathlib import Path

import pytest

from papercut.cli.main import main

from .conftest import make_pdf


def test_sources_list_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["sources", "list"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "tabme_pp" in captured.out
    assert "eurlex" in captured.out


def test_streams_build_creates_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pdf_dir = tmp_path / "pdfs"
    out_dir = tmp_path / "streams"
    make_pdf(pdf_dir / "a.pdf", 2)
    make_pdf(pdf_dir / "b.pdf", 1)
    make_pdf(pdf_dir / "c.pdf", 3)

    rc = main(
        [
            "streams",
            "build",
            str(pdf_dir),
            str(out_dir),
            "--n-streams",
            "4",
            "--mean-docs",
            "2",
            "--seed",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "Wrote 4 streams" in captured.out
    assert sum(1 for _ in out_dir.glob("stream_*.pdf")) == 4


def test_streams_build_rejects_missing_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["streams", "build", str(tmp_path / "does-not-exist"), str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert rc == 2
    assert "not found" in captured.err


def test_eval_baseline_smoke_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["eval", "baseline-smoke"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "every_page_new_doc" in captured.out
    assert "never_split" in captured.out


def test_eval_prospective_smoke_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["eval", "prospective-smoke"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "page_f1" in captured.out
