from __future__ import annotations

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class PageRef:
    """A page in a source document.

    `source` is a stable identifier (e.g. `"tabme/ffyw0199"`) so streams stay
    valid even when files move on disk; `page` is 0-indexed within that source.
    """

    source: str
    page: int


@dataclass(frozen=True, slots=True)
class Stream:
    """An ordered sequence of pages produced by one scan session.

    Convention: `boundaries[i] is True` iff page `i` begins a new document.
    Page 0 always begins a document, so `boundaries[0]` is always True when
    boundaries are known. `boundaries is None` marks an unlabeled stream.
    """

    pages: tuple[PageRef, ...]
    boundaries: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        if not self.pages:
            raise ValueError("Stream must contain at least one page")
        if self.boundaries is None:
            return
        if len(self.boundaries) != len(self.pages):
            raise ValueError(
                f"boundaries length {len(self.boundaries)} != pages length {len(self.pages)}"
            )
        if not self.boundaries[0]:
            raise ValueError("boundaries[0] must be True (first page starts first doc)")

    def __len__(self) -> int:
        return len(self.pages)

    @property
    def n_documents(self) -> int:
        if self.boundaries is None:
            raise ValueError("Stream is unlabeled")
        return sum(self.boundaries)

    def with_boundaries(self, boundaries: tuple[bool, ...]) -> Self:
        return type(self)(pages=self.pages, boundaries=boundaries)
