from __future__ import annotations

from pathlib import Path

import pytest

from papercut.streams.resolver import DictResolver, PdfDirectoryResolver, texts_for
from papercut.streams.types import PageRef

from .conftest import make_pdf


def test_pdf_directory_resolver_reads_pages(tmp_path: Path) -> None:
    make_pdf(tmp_path / "doc.pdf", 3, text="alpha")
    resolver = PdfDirectoryResolver(tmp_path)
    assert "alpha" in resolver.text(PageRef("doc", 0))
    assert "alpha" in resolver.text(PageRef("doc", 2))


def test_pdf_directory_resolver_caches_reader(tmp_path: Path) -> None:
    make_pdf(tmp_path / "doc.pdf", 2)
    resolver = PdfDirectoryResolver(tmp_path)
    resolver.text(PageRef("doc", 0))
    first = resolver._readers["doc"]
    resolver.text(PageRef("doc", 1))
    assert resolver._readers["doc"] is first


def test_pdf_directory_resolver_missing_source(tmp_path: Path) -> None:
    resolver = PdfDirectoryResolver(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolver.text(PageRef("missing", 0))


def test_pdf_directory_resolver_out_of_range(tmp_path: Path) -> None:
    make_pdf(tmp_path / "doc.pdf", 1)
    resolver = PdfDirectoryResolver(tmp_path)
    with pytest.raises(IndexError):
        resolver.text(PageRef("doc", 5))


def test_dict_resolver_roundtrips() -> None:
    p0, p1 = PageRef("x", 0), PageRef("x", 1)
    resolver = DictResolver({p0: "hello", p1: "world"})
    assert texts_for(resolver, [p0, p1]) == ["hello", "world"]


def test_dict_resolver_raises_on_missing() -> None:
    resolver = DictResolver({})
    with pytest.raises(KeyError):
        resolver.text(PageRef("nope", 0))
