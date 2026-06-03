from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from papercut.streams.resolver import PageResolver
from papercut.streams.types import PageRef, Stream


@dataclass(frozen=True)
class HfPssSchema:
    """Column mapping for an HF dataset where each row is one labeled stream.

    Most PSS datasets on the Hub follow a row-is-a-stream layout, with parallel
    arrays describing the pages and a boundary indicator. The exact column
    names differ between releases; instantiate `HfPssSchema` once per source.

    Either `page_ids_col` (stable per-page identifiers) or boundaries length
    alone determines page identity. `text_col` is optional: if present it
    yields per-page text for the resolver; if absent the corpus is image-only.

    `page_ids_col` values must be unique within the namespace because the
    corpus uses them as keys for its text cache. Sources where IDs only
    happen to repeat across streams should either disambiguate upstream or
    provide stream-prefixed IDs.
    """

    boundaries_col: str
    stream_id_col: str
    page_ids_col: str | None = None
    text_col: str | None = None


@dataclass
class HfPssCorpus:
    """In-memory corpus of `Stream`s with an attached text resolver.

    Acts as a `PageResolver` so models can be evaluated directly against it
    without rewriting prediction code per data backend.
    """

    streams: list[Stream]
    _texts: dict[PageRef, str] = field(default_factory=dict)
    _layouts: dict[PageRef, list[float]] = field(default_factory=dict)

    def text(self, page: PageRef) -> str:
        if page not in self._texts:
            raise KeyError(f"No text for {page}; corpus may be image-only")
        return self._texts[page]

    def layout(self, page: PageRef) -> list[float]:
        if page not in self._layouts:
            raise KeyError(f"No layout for {page}; corpus has no layout features")
        return self._layouts[page]

    def has_layouts(self) -> bool:
        return bool(self._layouts)

    def save(self, path: str | Path) -> None:
        """Pickle the corpus to disk for reuse.

        Streams plus text cache (and layouts if present); format is internal
        and unstable. Use this for caching a downloaded slice across dev
        sessions, not for sharing.
        """
        import pickle

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            pickle.dump(
                {
                    "streams": self.streams,
                    "texts": self._texts,
                    "layouts": self._layouts,
                },
                f,
            )

    @classmethod
    def load_from_disk(cls, path: str | Path) -> HfPssCorpus:
        """Inverse of `save`."""
        import pickle

        with Path(path).open("rb") as f:
            state = pickle.load(f)
        return cls(
            streams=state["streams"],
            _texts=state["texts"],
            _layouts=state.get("layouts", {}),
        )


def _row_to_stream(
    row: Mapping[str, Any],
    schema: HfPssSchema,
    source_namespace: str,
) -> tuple[Stream, list[tuple[PageRef, str]]]:
    raw_boundaries = row[schema.boundaries_col]
    boundaries = tuple(bool(b) for b in raw_boundaries)
    if not boundaries:
        raise ValueError("Empty boundaries vector")

    stream_id = str(row[schema.stream_id_col])

    if schema.page_ids_col is not None:
        ids: list[str] = [str(pid) for pid in row[schema.page_ids_col]]
    else:
        ids = [f"{stream_id}/{i}" for i in range(len(boundaries))]
    if len(ids) != len(boundaries):
        raise ValueError(
            f"page_ids length {len(ids)} != boundaries length {len(boundaries)} "
            f"for stream {stream_id}"
        )

    pages = tuple(PageRef(source=f"{source_namespace}:{pid}", page=0) for pid in ids)
    stream = Stream(pages=pages, boundaries=boundaries)

    text_pairs: list[tuple[PageRef, str]] = []
    if schema.text_col is not None:
        texts = row[schema.text_col]
        if len(texts) != len(pages):
            raise ValueError(
                f"text length {len(texts)} != boundaries length {len(boundaries)} "
                f"for stream {stream_id}"
            )
        text_pairs = list(zip(pages, [str(t) for t in texts], strict=True))

    return stream, text_pairs


def corpus_from_rows(
    rows: Iterable[Mapping[str, Any]],
    schema: HfPssSchema,
    source_namespace: str,
    max_streams: int | None = None,
) -> HfPssCorpus:
    """Build an `HfPssCorpus` from an iterable of row dicts.

    `rows` can be any iterable of mapping-like objects (an HF `Dataset`, a
    streaming iterator, or a list of fixture dicts in tests).
    """
    streams: list[Stream] = []
    texts: dict[PageRef, str] = {}
    for i, row in enumerate(rows):
        if max_streams is not None and i >= max_streams:
            break
        stream, pairs = _row_to_stream(row, schema, source_namespace)
        streams.append(stream)
        texts.update(pairs)
    if not streams:
        raise ValueError("No streams loaded; check schema and source")
    return HfPssCorpus(streams=streams, _texts=texts)


def load_hf_pss(
    repo_id: str,
    schema: HfPssSchema,
    *,
    split: str = "train",
    source_namespace: str | None = None,
    max_streams: int | None = None,
    streaming: bool = True,
) -> HfPssCorpus:
    """Pull a PSS dataset from the HuggingFace Hub and adapt it.

    Requires the `hf` optional extra. When `streaming=True` only the rows we
    actually consume are downloaded, which is the right default for dev work
    on a small slice; flip it off once we want the full corpus cached locally.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HF dataset loading needs the 'hf' extra: `uv sync --extra hf`") from e

    namespace = source_namespace or repo_id.split("/")[-1]
    rows: Iterable[Mapping[str, Any]] = load_dataset(repo_id, split=split, streaming=streaming)
    return corpus_from_rows(rows, schema, namespace, max_streams=max_streams)


def assert_resolver_protocol(c: HfPssCorpus) -> PageResolver:
    """Static-check that HfPssCorpus satisfies the PageResolver protocol."""
    return c
