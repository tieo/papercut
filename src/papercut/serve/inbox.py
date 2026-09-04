"""A non-destructive polling inbox for Paperless-ngx integration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from papercut.serve.predict import PdfSplitter, SplitReport


@dataclass(frozen=True)
class InboxResult:
    """A source stack processed in one inbox scan."""

    source: Path
    report: SplitReport


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
    ):
        raise ValueError(f"Invalid inbox state: {path}")
    return raw


def _save_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def process_inbox_once(
    inbox_dir: Path,
    consume_dir: Path,
    state_path: Path,
    splitter: PdfSplitter,
) -> tuple[InboxResult, ...]:
    """Split each unseen PDF in an inbox without deleting its source file."""
    if not inbox_dir.is_dir():
        raise ValueError(f"Inbox directory not found: {inbox_dir}")
    consume_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(state_path)
    results: list[InboxResult] = []
    for source in sorted(inbox_dir.glob("*.pdf")):
        digest = _sha256(source)
        key = str(source.resolve())
        if state.get(key) == digest:
            continue
        report = splitter.split(source, consume_dir, filename_prefix=f"{source.stem}-{digest[:12]}")
        state[key] = digest
        _save_state(state_path, state)
        results.append(InboxResult(source=source, report=report))
    return tuple(results)
