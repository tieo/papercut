from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from papercut.serve.inbox import process_inbox_once


@dataclass
class _Report:
    outputs: tuple[object, ...] = ()
    blank_pages: tuple[bool, ...] = ()


class _Splitter:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, str | None]] = []

    def split(self, source: Path, consume_dir: Path, filename_prefix: str | None = None) -> _Report:
        self.calls.append((source, consume_dir, filename_prefix))
        return _Report()


def test_process_inbox_once_leaves_sources_and_records_content_hash(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    source = inbox / "scan.pdf"
    source.write_bytes(b"first")
    consume = tmp_path / "consume"
    state = tmp_path / "state" / "processed.json"
    splitter = _Splitter()

    first = process_inbox_once(inbox, consume, state, splitter)
    second = process_inbox_once(inbox, consume, state, splitter)

    assert [result.source for result in first] == [source]
    assert second == ()
    assert source.read_bytes() == b"first"
    assert splitter.calls[0][2] == "scan-a7937b64b8ca"


def test_process_inbox_once_reprocesses_changed_input(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    source = inbox / "scan.pdf"
    source.write_bytes(b"first")
    splitter = _Splitter()
    state = tmp_path / "state.json"

    process_inbox_once(inbox, tmp_path / "consume", state, splitter)
    source.write_bytes(b"second")
    process_inbox_once(inbox, tmp_path / "consume", state, splitter)

    assert len(splitter.calls) == 2
