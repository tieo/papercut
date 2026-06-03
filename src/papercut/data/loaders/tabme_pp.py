from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from papercut.data.loaders.hf import HfPssCorpus
from papercut.streams.types import PageRef, Stream

TABME_REPO_ID = "rootsautomation/TABMEpp"


def parse_folders_file(path: Path) -> dict[int, list[str]]:
    """Parse a TABME++ `streams/<split>_folders.txt` into stream_id -> doc_ids.

    Each line is `<doc_id> <stream_id>`; lines belonging to the same stream
    are grouped, preserving the original order within the file (which is the
    intended stream order).
    """
    streams: dict[int, list[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            doc_id, stream_id_str = parts
            streams.setdefault(int(stream_id_str), []).append(doc_id)
    return streams


def extract_text_from_ocr(ocr_payload: str) -> str:
    """Pull a flat text string out of TABME++'s per-page OCR JSON.

    TABME++ stores each page's OCR as a JSON-encoded blob (Microsoft Azure
    Read API in modern releases). We walk the structure collecting every
    `text` field, which handles the Read API's nested
    analyzeResult/readResults/lines hierarchy as well as flatter variants.
    """
    if not ocr_payload:
        return ""
    try:
        data = json.loads(ocr_payload)
    except (TypeError, json.JSONDecodeError):
        return str(ocr_payload)

    parts: list[str] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, Mapping):
            text = obj.get("text")
            if isinstance(text, str):
                parts.append(text)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(data)
    return " ".join(parts)


def build_corpus(
    stream_defs: Mapping[int, list[str]],
    page_rows: Iterable[Mapping[str, Any]],
    *,
    max_streams: int | None = None,
) -> HfPssCorpus:
    """Compose an `HfPssCorpus` from stream definitions and raw page rows.

    Pure function with no network access: takes parsed inputs so it can be
    exercised by tests without a HuggingFace download.
    """
    selected_ids = sorted(stream_defs.keys())
    if max_streams is not None:
        selected_ids = selected_ids[:max_streams]
    needed_docs = {d for sid in selected_ids for d in stream_defs[sid]}

    pages_by_doc: dict[str, dict[int, str]] = {}
    for row in page_rows:
        doc_id = str(row["doc_id"])
        if doc_id not in needed_docs:
            continue
        text = extract_text_from_ocr(row.get("ocr", "") or "")
        pages_by_doc.setdefault(doc_id, {})[int(row["pg_id"])] = text

    streams: list[Stream] = []
    texts: dict[PageRef, str] = {}
    for sid in selected_ids:
        page_refs: list[PageRef] = []
        boundaries: list[bool] = []
        for doc_id in stream_defs[sid]:
            for pg_id, text in sorted(pages_by_doc.get(doc_id, {}).items()):
                ref = PageRef(source=f"tabme_pp/{doc_id}", page=pg_id)
                page_refs.append(ref)
                boundaries.append(pg_id == 0)
                texts[ref] = text
        if page_refs:
            streams.append(Stream(pages=tuple(page_refs), boundaries=tuple(boundaries)))
    if not streams:
        raise ValueError("No streams built; check that page rows cover the needed docs")
    return HfPssCorpus(streams=streams, _texts=texts)


def load(split: str = "train", max_streams: int | None = None) -> HfPssCorpus:
    """Download and adapt TABME++ for the given split.

    Requires the `hf` optional extra. Streams pages, so only the slice we ask
    for is materialized in memory; the whole 11GB corpus is fetched only if
    you ask for every stream.
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("TABME++ loading needs the 'hf' extra: `uv sync --extra hf`") from e

    folder_path = hf_hub_download(
        TABME_REPO_ID, f"streams/{split}_folders.txt", repo_type="dataset"
    )
    stream_defs = parse_folders_file(Path(folder_path))

    selected_ids = sorted(stream_defs.keys())
    if max_streams is not None:
        selected_ids = selected_ids[:max_streams]
    needed_docs = {d for sid in selected_ids for d in stream_defs[sid]}

    ds = load_dataset(TABME_REPO_ID, split=split, streaming=True)
    relevant_rows = (row for row in ds if row["doc_id"] in needed_docs)
    return build_corpus({sid: stream_defs[sid] for sid in selected_ids}, relevant_rows)
