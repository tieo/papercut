"""List all registered data sources. Usage: `uv run python scripts/list_sources.py`."""

from __future__ import annotations

import papercut.data.sources  # noqa: F401  (side effect: registers sources)
from papercut.data.manifest import REGISTRY


def main() -> None:
    if not REGISTRY:
        print("(no sources registered)")
        return
    for source in REGISTRY.values():
        langs = ",".join(source.languages)
        pages = f"{source.pages_estimate:,}" if source.pages_estimate else "?"
        print(f"{source.name:20s}  {langs:10s}  {pages:>12s} pages  {source.url}")


if __name__ == "__main__":
    main()
