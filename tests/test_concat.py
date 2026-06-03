from __future__ import annotations

import random
from pathlib import Path

import pytest
from pypdf import PdfReader

from papercut.streams.concat import (
    _poisson,
    build_stream_corpus,
    concat_pdfs,
    sample_stream_specs,
)
from papercut.streams.types import PageRef

from .conftest import make_pdf


def test_concat_pdfs_builds_correct_stream(tmp_path: Path) -> None:
    pdf_a = make_pdf(tmp_path / "a.pdf", 3)
    pdf_b = make_pdf(tmp_path / "b.pdf", 1)
    pdf_c = make_pdf(tmp_path / "c.pdf", 2)

    stream = concat_pdfs(
        [("a", pdf_a), ("b", pdf_b), ("c", pdf_c)],
        tmp_path / "merged.pdf",
    )

    assert len(stream) == 6
    assert stream.boundaries == (True, False, False, True, True, False)
    assert stream.pages[0] == PageRef(source="a", page=0)
    assert stream.pages[2] == PageRef(source="a", page=2)
    assert stream.pages[3] == PageRef(source="b", page=0)
    assert stream.pages[4] == PageRef(source="c", page=0)
    assert stream.n_documents == 3

    merged = PdfReader(str(tmp_path / "merged.pdf"))
    assert len(merged.pages) == 6


def test_concat_pdfs_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty source list"):
        concat_pdfs([], tmp_path / "out.pdf")


def test_concat_pdfs_rejects_zero_page_source(tmp_path: Path) -> None:
    empty = make_pdf(tmp_path / "empty.pdf", 0)
    with pytest.raises(ValueError, match="zero pages"):
        concat_pdfs([("empty", empty)], tmp_path / "out.pdf")


def test_poisson_distribution_shape() -> None:
    rng = random.Random(42)
    samples = [_poisson(rng, 5.0) for _ in range(2000)]
    mean = sum(samples) / len(samples)
    assert 4.5 < mean < 5.5
    assert min(samples) >= 0


def test_sample_stream_specs_is_deterministic_per_seed() -> None:
    pool = [(f"s{i}", Path(f"/tmp/s{i}.pdf")) for i in range(20)]
    specs_a = sample_stream_specs(pool, n_streams=5, mean_docs_per_stream=4.0, rng=random.Random(7))
    specs_b = sample_stream_specs(pool, n_streams=5, mean_docs_per_stream=4.0, rng=random.Random(7))
    assert specs_a == specs_b


def test_sample_stream_specs_respects_pool_size() -> None:
    pool = [(f"s{i}", Path(f"/tmp/s{i}.pdf")) for i in range(3)]
    rng = random.Random(0)
    specs = sample_stream_specs(pool, n_streams=10, mean_docs_per_stream=20.0, rng=rng)
    for spec in specs:
        assert 1 <= len(spec) <= 3
        ids = [s[0] for s in spec]
        assert len(set(ids)) == len(ids)


def test_build_stream_corpus_writes_files(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "sources"
    pool = [
        ("doc1", make_pdf(pdf_dir / "doc1.pdf", 3)),
        ("doc2", make_pdf(pdf_dir / "doc2.pdf", 2)),
        ("doc3", make_pdf(pdf_dir / "doc3.pdf", 1)),
        ("doc4", make_pdf(pdf_dir / "doc4.pdf", 4)),
    ]
    out_dir = tmp_path / "streams"
    streams = build_stream_corpus(
        pool,
        out_dir,
        n_streams=5,
        mean_docs_per_stream=2.5,
        seed=0,
    )
    assert len(streams) == 5
    for i, stream in enumerate(streams):
        assert (out_dir / f"stream_{i:06d}.pdf").exists()
        assert stream.boundaries is not None
        assert stream.boundaries[0] is True
        assert stream.n_documents == sum(stream.boundaries)
