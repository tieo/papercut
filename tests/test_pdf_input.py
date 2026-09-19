from __future__ import annotations

import math
from pathlib import Path

import pytest

from papercut.data.loaders.tabme_pp import LAYOUT_FEATURE_NAMES
from papercut.serve.pdf_input import parse_tesseract_tsv


def test_parse_tesseract_tsv_uses_normalized_word_boxes() -> None:
    tsv = "\n".join(
        [
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
            "5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t95\tHello",
            "5\t1\t1\t1\t1\t2\t50\t20\t20\t10\t95\tworld",
        ]
    )
    text, layout = parse_tesseract_tsv(tsv, image_width=100, image_height=100)

    assert text == "Hello world"
    assert len(layout) == len(LAYOUT_FEATURE_NAMES)
    assert layout[0] == pytest.approx(math.log1p(2))
    assert layout[7] == pytest.approx(0.25)
    assert layout[9] == pytest.approx(0.425)


def test_parse_tesseract_tsv_rejects_invalid_image_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        parse_tesseract_tsv("", image_width=0, image_height=100)


def test_detect_rotation_reads_the_osd_angle(monkeypatch) -> None:
    import subprocess

    from papercut.serve import pdf_input as module

    def fake_run(args, **kwargs):
        assert "--psm" in args and args[args.index("--psm") + 1] == "0"
        return subprocess.CompletedProcess(args, 0, stdout="Page number: 0\nRotate: 180\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.detect_rotation(Path("page.png"), "tesseract") == 180


def test_detect_rotation_falls_back_when_osd_fails(monkeypatch) -> None:
    import subprocess

    from papercut.serve import pdf_input as module

    def failing_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="Too few characters")

    monkeypatch.setattr(module.subprocess, "run", failing_run)
    assert module.detect_rotation(Path("blank.png"), "tesseract") == 0
