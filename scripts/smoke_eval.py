"""End-to-end smoke: evaluate trivial baselines on hand-built streams.

Usage: `uv run python scripts/smoke_eval.py`
"""

from __future__ import annotations

from papercut.eval.runner import evaluate
from papercut.models.baselines.trivial import EveryPageNewDoc, NeverSplit
from papercut.streams.types import PageRef, Stream


def fixture_streams() -> list[Stream]:
    cases: list[tuple[bool, ...]] = [
        (True, False, False, True, False),
        (True, False, True, False, False, False),
        (True, False, False, False, False, False),
        (True,),
        (True, True, True),
    ]
    out: list[Stream] = []
    for i, bnd in enumerate(cases):
        pages = tuple(PageRef(source=f"fixture/{i}", page=p) for p in range(len(bnd)))
        out.append(Stream(pages=pages, boundaries=bnd))
    return out


def main() -> None:
    streams = fixture_streams()
    for model in [EveryPageNewDoc(), NeverSplit()]:
        report = evaluate(model, streams)
        print(
            f"{model.name:24s}  "
            f"page_f1={report.page_f1_mean:.3f}  "
            f"pq={report.pq_mean:.3f}  "
            f"stp={report.stp:.3f}  "
            f"mndd_mean={report.mndd_mean:.2f}  "
            f"(n={report.n_streams})"
        )


if __name__ == "__main__":
    main()
