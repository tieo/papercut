from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path


def _cmd_sources_list(_: argparse.Namespace) -> int:
    import papercut.data.sources  # noqa: F401 (registers sources)
    from papercut.data.manifest import REGISTRY

    if not REGISTRY:
        print("(no sources registered)")
        return 0
    for source in REGISTRY.values():
        langs = ",".join(source.languages)
        pages = f"{source.pages_estimate:,}" if source.pages_estimate else "?"
        print(f"{source.name:24s}  {langs:10s}  {pages:>12s}  {source.url}")
    return 0


def _cmd_streams_build(args: argparse.Namespace) -> int:
    from papercut.streams.concat import build_stream_corpus

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    if not pdf_dir.is_dir():
        print(f"PDF dir not found: {pdf_dir}", file=sys.stderr)
        return 2
    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs under {pdf_dir}", file=sys.stderr)
        return 2

    pool = [(str(p.relative_to(pdf_dir).with_suffix("")), p) for p in pdfs]
    streams = build_stream_corpus(
        pool,
        out_dir,
        n_streams=args.n_streams,
        mean_docs_per_stream=args.mean_docs,
        seed=args.seed,
    )
    print(f"Wrote {len(streams)} streams to {out_dir}")
    return 0


def _cmd_eval_baseline_smoke(_: argparse.Namespace) -> int:
    from papercut.eval.runner import evaluate
    from papercut.models.baselines.trivial import EveryPageNewDoc, NeverSplit
    from papercut.streams.types import PageRef, Stream

    rng = random.Random(0)
    streams: list[Stream] = []
    for _i in range(5):
        n = rng.randint(2, 8)
        boundaries = [True] + [rng.random() < 0.3 for _ in range(n - 1)]
        pages = tuple(PageRef(source=f"fx{_i}", page=p) for p in range(n))
        streams.append(Stream(pages=pages, boundaries=tuple(boundaries)))

    for model in [EveryPageNewDoc(), NeverSplit()]:
        report = evaluate(model, streams)
        print(
            f"{model.name:24s}  page_f1={report.page_f1_mean:.3f}  "
            f"pq={report.pq_mean:.3f}  stp={report.stp:.3f}  "
            f"mndd_mean={report.mndd_mean:.2f}"
        )
    return 0


def _cmd_data_download_tabme(args: argparse.Namespace) -> int:
    from papercut.data.loaders import tabme_pp

    out_path = Path(args.out)
    print(f"Downloading TABME++ {args.split} slice (max_streams={args.max_streams}) -> {out_path}")
    corpus = tabme_pp.load(split=args.split, max_streams=args.max_streams)
    corpus.save(out_path)
    print(
        f"Saved {len(corpus.streams)} streams "
        f"({sum(len(s) for s in corpus.streams)} pages) to {out_path}"
    )
    return 0


def _cmd_eval_prospective_smoke(_: argparse.Namespace) -> int:
    from papercut.eval.prospective import Slice, format_results, walk_forward
    from papercut.models.baselines.trivial import EveryPageNewDoc, NeverSplit
    from papercut.streams.types import PageRef, Stream

    def _s(boundaries: tuple[bool, ...]) -> Stream:
        pages = tuple(PageRef(source="fx", page=i) for i in range(len(boundaries)))
        return Stream(pages=pages, boundaries=boundaries)

    slices = [
        Slice("2024-Q1", [_s((True, False, False, True, False))]),
        Slice("2024-Q2", [_s((True, False, True, False, False, False))]),
        Slice("2024-Q3", [_s((True, False, False, True, True, False))]),
    ]
    results = walk_forward([EveryPageNewDoc(), NeverSplit()], slices)
    print(format_results(results))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="papercut", description="Page Stream Segmentation toolkit."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sources = sub.add_parser("sources", help="Inspect the data source registry.")
    sources_sub = sources.add_subparsers(dest="sources_command", required=True)
    sources_sub.add_parser("list", help="List all registered sources.").set_defaults(
        func=_cmd_sources_list
    )

    data = sub.add_parser("data", help="Download and cache external datasets.")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    download = data_sub.add_parser("download", help="Download a registered source slice.")
    download_sub = download.add_subparsers(dest="download_source", required=True)
    tabme = download_sub.add_parser("tabme-pp", help="Pull a TABME++ slice from HuggingFace.")
    tabme.add_argument("--split", choices=["train", "val", "test"], default="test")
    tabme.add_argument("--max-streams", type=int, default=50)
    tabme.add_argument("--out", default="data/tabme_pp_slice.pkl")
    tabme.set_defaults(func=_cmd_data_download_tabme)

    streams = sub.add_parser("streams", help="Build labeled stream corpora.")
    streams_sub = streams.add_subparsers(dest="streams_command", required=True)
    build = streams_sub.add_parser(
        "build", help="Concatenate single-doc PDFs into labeled streams."
    )
    build.add_argument("pdf_dir", help="Directory holding single-document PDFs (recursive).")
    build.add_argument("out_dir", help="Where to write merged stream PDFs.")
    build.add_argument("--n-streams", type=int, default=100)
    build.add_argument("--mean-docs", type=float, default=10.0)
    build.add_argument("--seed", type=int, default=0)
    build.set_defaults(func=_cmd_streams_build)

    eval_p = sub.add_parser("eval", help="Run evaluation.")
    eval_sub = eval_p.add_subparsers(dest="eval_command", required=True)
    eval_sub.add_parser(
        "baseline-smoke", help="Run trivial baselines on synthetic fixtures."
    ).set_defaults(func=_cmd_eval_baseline_smoke)
    eval_sub.add_parser(
        "prospective-smoke", help="Run a walk-forward demo on synthetic slices."
    ).set_defaults(func=_cmd_eval_prospective_smoke)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
