from __future__ import annotations

from dataclasses import dataclass

from papercut.streams.types import Stream


@dataclass(frozen=True)
class EveryPageNewDoc:
    """Predict that every page starts a new document.

    Degenerate baseline. Hits perfect recall on boundary class with precision
    equal to the true boundary rate. Useful as an upper bound on recall and a
    floor for any model that claims to learn anything.
    """

    name: str = "every_page_new_doc"

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        return tuple([True] * len(stream))


@dataclass(frozen=True)
class NeverSplit:
    """Predict that the entire stream is a single document.

    Degenerate baseline. Hits zero recall but precision is undefined (no
    positive predictions). On corpora dominated by long documents this can
    score deceptively well on accuracy-style metrics, which is exactly why
    PSS leans on STP and Panoptic Quality instead.
    """

    name: str = "never_split"

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        return tuple([True] + [False] * (len(stream) - 1))
