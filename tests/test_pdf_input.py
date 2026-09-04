from __future__ import annotations

import math

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
