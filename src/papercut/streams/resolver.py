from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from pypdf import PdfReader

from papercut.streams.types import PageRef


@runtime_checkable
class PageResolver(Protocol):
    """Resolves a `PageRef` to its rendered content.

    Models that need to look at page content depend on a resolver rather than
    on raw file paths, so the same model code can serve a real PDF backend, a
    pre-extracted text cache, or a test fixture.
    """

    def text(self, page: PageRef) -> str: ...


class PdfDirectoryResolver:
    """Resolves `PageRef.source` against `<root>/<source>.pdf` files.

    Caches one `PdfReader` per source path. The resolver assumes the source
    identifier maps to a file under `root` with a `.pdf` extension; if the
    source identifier already includes the extension or contains slashes, it
    is treated as a relative path.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._readers: dict[str, PdfReader] = {}

    def _path_for(self, source: str) -> Path:
        candidate = (self.root / source).with_suffix(".pdf")
        if candidate.exists():
            return candidate
        candidate = self.root / source
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No PDF for source {source!r} under {self.root}")

    def _reader(self, source: str) -> PdfReader:
        if source not in self._readers:
            self._readers[source] = PdfReader(str(self._path_for(source)))
        return self._readers[source]

    def text(self, page: PageRef) -> str:
        reader = self._reader(page.source)
        if page.page >= len(reader.pages):
            raise IndexError(
                f"Source {page.source!r} has {len(reader.pages)} pages; asked for page {page.page}"
            )
        return reader.pages[page.page].extract_text() or ""


class DictResolver:
    """In-memory `PageResolver` backed by a dict, for tests."""

    def __init__(self, texts: dict[PageRef, str]) -> None:
        self._texts = texts

    def text(self, page: PageRef) -> str:
        if page not in self._texts:
            raise KeyError(page)
        return self._texts[page]


def texts_for(resolver: PageResolver, pages: Iterable[PageRef]) -> list[str]:
    return [resolver.text(p) for p in pages]
