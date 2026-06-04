from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from papercut.data.loaders.hf import HfPssCorpus
from papercut.streams.types import PageRef, Stream

TABME_REPO_ID = "rootsautomation/TABMEpp"
VISION_FEATURE_NAMES = (
    "aspect_ratio",
    "mean_intensity",
    "std_intensity",
    "edge_density",
    "intensity_top",
    "intensity_bottom",
    "intensity_left",
    "intensity_right",
    "intensity_q11",
    "intensity_q12",
    "intensity_q13",
    "intensity_q21",
    "intensity_q22",
    "intensity_q23",
    "intensity_q31",
    "intensity_q32",
    "intensity_q33",
)


def extract_visual_from_img(img_bytes: bytes | None) -> list[float]:
    """Cheap per-page visual statistics from raw image bytes.

    Decodes the page, converts to grayscale, resizes to 96x96, and computes
    aspect ratio, mean and std intensity, a Sobel-like edge density proxy,
    plus mean intensity in 4 horizontal halves and a 3x3 spatial grid. The
    grid intensities capture letterhead, address-block, and signature-block
    presence at coarse resolution without pulling in a CNN.
    """
    n = len(VISION_FEATURE_NAMES)
    if not img_bytes:
        return [0.0] * n
    try:
        import io

        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        w, h = img.size
        aspect = float(w) / max(1, h)
        small = img.resize((96, 96), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0
        mean_i = float(arr.mean())
        std_i = float(arr.std())
        # Edge density: mean of absolute gradient
        gy = np.abs(arr[1:, :] - arr[:-1, :]).mean()
        gx = np.abs(arr[:, 1:] - arr[:, :-1]).mean()
        edge = float((gx + gy) / 2.0)
        # Horizontal halves and vertical halves
        top = float(arr[:48, :].mean())
        bot = float(arr[48:, :].mean())
        left = float(arr[:, :48].mean())
        right = float(arr[:, 48:].mean())
        # 3x3 grid
        grid: list[float] = []
        for i in range(3):
            for j in range(3):
                cell = arr[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32]
                grid.append(float(cell.mean()))
        return [aspect, mean_i, std_i, edge, top, bot, left, right, *grid]
    except Exception:
        return [0.0] * n


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


_TEXT_KEYS = ("Word", "text", "content")
LAYOUT_FEATURE_NAMES = (
    "n_words_log",
    "top_density",
    "bottom_density",
    "left_density",
    "right_density",
    "top_left_density",
    "center_band_density",
    "mean_y",
    "std_y",
    "mean_x",
    "y_min",
    "y_max",
    "bbox_area_coverage",
    "mean_word_height",
)


def extract_layout_from_ocr(ocr_payload: str) -> list[float]:
    """Compute per-page layout features from TABME++ OCR bbox JSON.

    Each `lines_data` entry has normalised quad coordinates (X1..4, Y1..4 in
    [0, 1]). We derive zero-cost layout signals: header / footer / address-
    block density, vertical word distribution, mean word height. These give
    the model letterhead and structural cues that pure TF-IDF misses,
    without rendering images or pulling in a vision model.
    """
    n = len(LAYOUT_FEATURE_NAMES)
    if not ocr_payload:
        return [0.0] * n
    try:
        data = json.loads(ocr_payload)
    except (TypeError, json.JSONDecodeError):
        return [0.0] * n

    words: list[dict[str, Any]] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, Mapping):
            if all(f"{axis}{idx}" in obj for axis in "XY" for idx in (1, 2, 3, 4)):
                words.append(obj)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(data)
    if not words:
        return [0.0] * n

    import math

    ys = [(float(w["Y1"]) + float(w["Y3"])) / 2 for w in words]
    xs = [(float(w["X1"]) + float(w["X3"])) / 2 for w in words]
    heights = [abs(float(w["Y3"]) - float(w["Y1"])) for w in words]
    widths = [abs(float(w["X3"]) - float(w["X1"])) for w in words]
    n_w = len(words)
    top = sum(1 for y in ys if y < 0.20) / n_w
    bottom = sum(1 for y in ys if y > 0.80) / n_w
    left = sum(1 for x in xs if x < 0.30) / n_w
    right = sum(1 for x in xs if x > 0.70) / n_w
    top_left = sum(1 for x, y in zip(xs, ys, strict=True) if x < 0.40 and y < 0.20) / n_w
    center_band = sum(1 for y in ys if 0.30 <= y <= 0.70) / n_w
    mean_y = sum(ys) / n_w
    mean_x = sum(xs) / n_w
    variance_y = sum((y - mean_y) ** 2 for y in ys) / n_w
    bbox_area = sum(h * w for h, w in zip(heights, widths, strict=True))
    mean_word_h = sum(heights) / n_w

    return [
        math.log1p(n_w),
        top,
        bottom,
        left,
        right,
        top_left,
        center_band,
        mean_y,
        math.sqrt(variance_y),
        mean_x,
        min(ys),
        max(ys),
        min(1.0, bbox_area),
        mean_word_h,
    ]


def extract_text_from_ocr(ocr_payload: str) -> str:
    """Pull a flat text string out of TABME++'s per-page OCR JSON.

    Empirically TABME++ wraps its per-line OCR as
    `{"lines_data": [{"Word": "<segment>", ...}, ...]}`, where each entry's
    `Word` field is actually a multi-word text segment (the field name is
    Roots Automation's, not ours). For robustness against other OCR shapes
    (Microsoft Azure Read API uses `text`, some custom formats use
    `content`) we walk recursively and collect any of those known keys.
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
            for key in _TEXT_KEYS:
                value = obj.get(key)
                if isinstance(value, str):
                    parts.append(value)
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

    pages_by_doc_text: dict[str, dict[int, str]] = {}
    pages_by_doc_layout: dict[str, dict[int, list[float]]] = {}
    pages_by_doc_visual: dict[str, dict[int, list[float]]] = {}
    for row in page_rows:
        doc_id = str(row["doc_id"])
        if doc_id not in needed_docs:
            continue
        ocr_str = row.get("ocr", "") or ""
        pages_by_doc_text.setdefault(doc_id, {})[int(row["pg_id"])] = extract_text_from_ocr(ocr_str)
        pages_by_doc_layout.setdefault(doc_id, {})[int(row["pg_id"])] = extract_layout_from_ocr(
            ocr_str
        )
        img_bytes = row.get("img")
        pages_by_doc_visual.setdefault(doc_id, {})[int(row["pg_id"])] = extract_visual_from_img(
            img_bytes
        )

    streams: list[Stream] = []
    texts: dict[PageRef, str] = {}
    layouts: dict[PageRef, list[float]] = {}
    visuals: dict[PageRef, list[float]] = {}
    for sid in selected_ids:
        page_refs: list[PageRef] = []
        boundaries: list[bool] = []
        for doc_id in stream_defs[sid]:
            doc_pages = pages_by_doc_text.get(doc_id, {})
            doc_layouts = pages_by_doc_layout.get(doc_id, {})
            doc_visuals = pages_by_doc_visual.get(doc_id, {})
            for pg_id, text in sorted(doc_pages.items()):
                ref = PageRef(source=f"tabme_pp/{doc_id}", page=pg_id)
                page_refs.append(ref)
                boundaries.append(pg_id == 0)
                texts[ref] = text
                layouts[ref] = doc_layouts.get(pg_id, [0.0] * len(LAYOUT_FEATURE_NAMES))
                visuals[ref] = doc_visuals.get(pg_id, [0.0] * len(VISION_FEATURE_NAMES))
        if page_refs:
            streams.append(Stream(pages=tuple(page_refs), boundaries=tuple(boundaries)))
    if not streams:
        raise ValueError("No streams built; check that page rows cover the needed docs")
    return HfPssCorpus(streams=streams, _texts=texts, _layouts=layouts, _visuals=visuals)


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
