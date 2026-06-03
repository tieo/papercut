from __future__ import annotations

import json
from pathlib import Path

import pytest

from papercut.data.loaders.tabme_pp import (
    build_corpus,
    extract_text_from_ocr,
    parse_folders_file,
)
from papercut.streams.types import PageRef


def test_parse_folders_file_groups_by_stream(tmp_path: Path) -> None:
    path = tmp_path / "folders.txt"
    path.write_text("doc_a 1\ndoc_b 1\ndoc_c 2\ndoc_d 2\ndoc_e 2\n")
    out = parse_folders_file(path)
    assert out == {1: ["doc_a", "doc_b"], 2: ["doc_c", "doc_d", "doc_e"]}


def test_parse_folders_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "folders.txt"
    path.write_text("doc_a 1\n\nmalformed line with extra\ndoc_b 2\n")
    out = parse_folders_file(path)
    assert out == {1: ["doc_a"], 2: ["doc_b"]}


def test_extract_text_from_microsoft_read_api_shape() -> None:
    payload = json.dumps(
        {
            "analyzeResult": {
                "readResults": [
                    {
                        "lines": [
                            {"text": "Hello world"},
                            {"text": "Second line"},
                        ]
                    }
                ]
            }
        }
    )
    assert extract_text_from_ocr(payload) == "Hello world Second line"


def test_extract_text_from_flat_lines() -> None:
    payload = json.dumps([{"text": "alpha"}, {"text": "beta"}])
    assert extract_text_from_ocr(payload) == "alpha beta"


def test_extract_text_from_tabme_pp_native_shape() -> None:
    """Empirical TABME++ shape: lines_data with Word per segment."""
    payload = json.dumps(
        {
            "lines_data": [
                {"Word": "TABLE 4", "Confidence": 0.99, "page": 0},
                {"Word": "CIGARETTES WITHOUT COUMARIN", "Confidence": 0.99, "page": 0},
            ]
        }
    )
    assert extract_text_from_ocr(payload) == "TABLE 4 CIGARETTES WITHOUT COUMARIN"


def test_extract_text_empty_and_invalid() -> None:
    assert extract_text_from_ocr("") == ""
    assert extract_text_from_ocr("not json") == "not json"


def test_build_corpus_orders_pages_by_pg_id() -> None:
    stream_defs = {1: ["doc_a", "doc_b"], 2: ["doc_c"]}
    rows = [
        {"doc_id": "doc_a", "pg_id": 1, "ocr": '{"text": "a1"}'},
        {"doc_id": "doc_a", "pg_id": 0, "ocr": '{"text": "a0"}'},
        {"doc_id": "doc_b", "pg_id": 0, "ocr": '{"text": "b0"}'},
        {"doc_id": "doc_c", "pg_id": 0, "ocr": '{"text": "c0"}'},
        {"doc_id": "doc_c", "pg_id": 1, "ocr": '{"text": "c1"}'},
    ]
    corpus = build_corpus(stream_defs, rows)
    assert len(corpus.streams) == 2
    s1 = corpus.streams[0]
    assert s1.boundaries == (True, False, True)
    assert s1.pages[0] == PageRef(source="tabme_pp/doc_a", page=0)
    assert corpus.text(s1.pages[0]) == "a0"
    assert corpus.text(s1.pages[1]) == "a1"
    assert corpus.text(s1.pages[2]) == "b0"


def test_build_corpus_respects_max_streams() -> None:
    stream_defs = {1: ["doc_a"], 2: ["doc_b"], 3: ["doc_c"]}
    rows = [
        {"doc_id": "doc_a", "pg_id": 0, "ocr": ""},
        {"doc_id": "doc_b", "pg_id": 0, "ocr": ""},
        {"doc_id": "doc_c", "pg_id": 0, "ocr": ""},
    ]
    corpus = build_corpus(stream_defs, rows, max_streams=2)
    assert len(corpus.streams) == 2


def test_build_corpus_skips_missing_pages() -> None:
    """A stream whose docs have no page rows present is silently dropped."""
    stream_defs = {1: ["present"], 2: ["missing"]}
    rows = [{"doc_id": "present", "pg_id": 0, "ocr": ""}]
    corpus = build_corpus(stream_defs, rows)
    assert len(corpus.streams) == 1


def test_build_corpus_raises_when_nothing_built() -> None:
    stream_defs = {1: ["missing"]}
    with pytest.raises(ValueError, match="No streams built"):
        build_corpus(stream_defs, [])
