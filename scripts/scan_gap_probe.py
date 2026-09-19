"""Measure how much a model loses when its pages come off a scanner.

The personal corpus is built from PDFs that carry their own text layer, so
rendering them produces clean glyphs and OCR reads them almost perfectly.
Real stacks arrive skewed, soft and speckled. This takes the same documents
twice, once rendered clean and once degraded, composes the same streams over
both, and evaluates the same model on each. The difference is the cost of the
scanner, separated from every other difference between the corpora.

Example:
    nix-shell --run '.venv/bin/python scripts/scan_gap_probe.py \\
        --source-dir data/private/nasx-paperless-originals \\
        --model data/models/layout_sem_vis_personal_only_6k.pkl --documents 20'
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from papercut.data.loaders.hf import HfPssCorpus
from papercut.eval.metrics import mndd, page_metrics, stp
from papercut.models.baselines.tfidf_xgb_layout_sem_vis import TfIdfXgbLayoutSemVis
from papercut.serve.pdf_input import pdf_input
from papercut.streams.compose import corpus_from_documents
from papercut.streams.types import PageRef, Stream


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--streams", type=int, default=200)
    parser.add_argument("--mean-documents", type=float, default=1.5)
    parser.add_argument("--languages", default="deu+eng")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def _corpus_for(paths: Sequence[Path], languages: str, degrade: bool, max_pages: int):
    texts: dict[PageRef, str] = {}
    layouts: dict[PageRef, list[float]] = {}
    visuals: dict[PageRef, list[float]] = {}
    documents = []
    for path in paths:
        identifier = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        parsed = pdf_input(
            path,
            languages=languages,
            degrade_seed=hash(identifier) % 10_000 if degrade else None,
        )
        source_pages = parsed.stream.pages
        if len(source_pages) > max_pages:
            continue
        pages = tuple(PageRef(source=f"probe/{identifier}", page=i) for i in range(len(source_pages)))
        for page, original in zip(pages, source_pages, strict=True):
            texts[page] = parsed.corpus.text(original)
            layouts[page] = parsed.corpus.layout(original)
            visuals[page] = parsed.corpus.visual(original)
        documents.append((f"probe/{identifier}", pages))
    corpus = HfPssCorpus(
        streams=[Stream(pages=pages, boundaries=(True, *([False] * (len(pages) - 1)))) for _, pages in documents],
        _texts=texts,
        _layouts=layouts,
        _visuals=visuals,
    )
    return corpus, documents


def _score(model_path: Path, corpus: HfPssCorpus) -> dict[str, float]:
    model = TfIdfXgbLayoutSemVis.load_with_corpus(model_path, corpus)
    pairs = [(stream.boundaries, model.predict_boundaries(stream)) for stream in corpus.streams]
    chars = [len(corpus.text(page)) for stream in corpus.streams for page in stream.pages]
    chars.sort()
    return {
        "stp": stp(pairs),
        "page_f1_mean": sum(page_metrics(t, p).f1 for t, p in pairs) / len(pairs),
        "mndd_mean": sum(mndd(t, p) for t, p in pairs) / len(pairs),
        "median_chars": chars[len(chars) // 2],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    paths = sorted(p for p in args.source_dir.rglob("*.pdf"))[: args.documents]
    if not paths:
        raise ValueError(f"No PDFs under {args.source_dir}")

    report = {}
    for label, degrade in (("clean", False), ("scanned", True)):
        page_corpus, documents = _corpus_for(paths, args.languages, degrade, args.max_pages)
        composed = corpus_from_documents(
            page_corpus,
            documents,
            n_streams=args.streams,
            mean_documents_per_stream=args.mean_documents,
            seed=args.seed,
        )
        report[label] = _score(args.model, composed)
        print(label, json.dumps(report[label], sort_keys=True), flush=True)

    print(json.dumps({"model": str(args.model), "documents": len(paths), **report}, indent=2, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
