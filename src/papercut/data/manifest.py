from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Modality(StrEnum):
    IMAGE = "image"
    TEXT = "text"
    IMAGE_TEXT = "image+text"


class Labeling(StrEnum):
    PSS_LABELED = "pss_labeled"
    SINGLE_DOC_PDFS = "single_doc_pdfs"
    BULK_CORPUS = "bulk_corpus"


class Source(BaseModel):
    """One dataset we can pull pages from."""

    name: str
    languages: list[str] = Field(description="ISO 639-1 codes; e.g. ['en', 'de']")
    modality: Modality
    labeling: Labeling
    pages_estimate: int | None = None
    license: str | None = None
    url: str
    notes: str | None = None


REGISTRY: dict[str, Source] = {}


def register(source: Source) -> Source:
    if source.name in REGISTRY:
        raise ValueError(f"Source {source.name!r} already registered")
    REGISTRY[source.name] = source
    return source
