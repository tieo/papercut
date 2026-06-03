"""Prospective evaluation demo with trivial baselines on fixture slices.

Usage: `uv run python scripts/prospective_smoke.py`
"""

from __future__ import annotations

from papercut.eval.prospective import Slice, format_results, walk_forward
from papercut.models.baselines.trivial import EveryPageNewDoc, NeverSplit
from papercut.streams.types import PageRef, Stream


def _stream(boundaries: tuple[bool, ...]) -> Stream:
    pages = tuple(PageRef(source="fx", page=i) for i in range(len(boundaries)))
    return Stream(pages=pages, boundaries=boundaries)


def main() -> None:
    slices = [
        Slice(
            "2024-Q1",
            [
                _stream((True, False, False, True, False)),
                _stream((True, False, True, False, False, False)),
            ],
        ),
        Slice(
            "2024-Q2",
            [
                _stream((True,)),
                _stream((True, True)),
                _stream((True, False, False, False, True, False)),
            ],
        ),
        Slice(
            "2024-Q3",
            [
                _stream((True, False, True, False, False)),
                _stream((True, False, False)),
            ],
        ),
    ]
    results = walk_forward([EveryPageNewDoc(), NeverSplit()], slices)
    print(format_results(results))


if __name__ == "__main__":
    main()
