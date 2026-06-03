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


def _build_model(name: str, resolver: object) -> object:
    if name == "trivial:every-page":
        from papercut.models.baselines.trivial import EveryPageNewDoc

        return EveryPageNewDoc()
    if name == "trivial:never-split":
        from papercut.models.baselines.trivial import NeverSplit

        return NeverSplit()
    if name == "text-similarity":
        from papercut.models.baselines.text_similarity import TextSimilarityBaseline

        return TextSimilarityBaseline(resolver=resolver)  # type: ignore[arg-type]
    if name == "tfidf-xgb":
        from papercut.models.baselines.tfidf_xgb import TfIdfXgb

        return TfIdfXgb(resolver=resolver)  # type: ignore[arg-type]
    if name == "ensemble:text-sim+tfidf-xgb":
        from papercut.models.baselines.text_similarity import TextSimilarityBaseline
        from papercut.models.baselines.tfidf_xgb import TfIdfXgb
        from papercut.models.ensembles.late import LateEnsemble

        return LateEnsemble(
            submodels=[
                TextSimilarityBaseline(resolver=resolver),  # type: ignore[arg-type]
                TfIdfXgb(resolver=resolver),  # type: ignore[arg-type]
            ],
            name="ensemble:text-sim+tfidf-xgb",
        )
    if name == "viterbi:text-similarity":
        from papercut.models.baselines.text_similarity import TextSimilarityBaseline
        from papercut.models.smoothing.viterbi import SequenceSmoothed

        return SequenceSmoothed(
            submodel=TextSimilarityBaseline(resolver=resolver),  # type: ignore[arg-type]
            name="viterbi:text-similarity",
        )
    if name == "viterbi:tfidf-xgb":
        from papercut.models.baselines.tfidf_xgb import TfIdfXgb
        from papercut.models.smoothing.viterbi import SequenceSmoothed

        return SequenceSmoothed(
            submodel=TfIdfXgb(resolver=resolver),  # type: ignore[arg-type]
            name="viterbi:tfidf-xgb",
        )
    if name == "viterbi:ensemble":
        from papercut.models.baselines.text_similarity import TextSimilarityBaseline
        from papercut.models.baselines.tfidf_xgb import TfIdfXgb
        from papercut.models.ensembles.late import LateEnsemble
        from papercut.models.smoothing.viterbi import SequenceSmoothed

        return SequenceSmoothed(
            submodel=LateEnsemble(
                submodels=[
                    TextSimilarityBaseline(resolver=resolver),  # type: ignore[arg-type]
                    TfIdfXgb(resolver=resolver),  # type: ignore[arg-type]
                ],
            ),
            name="viterbi:ensemble",
        )
    raise ValueError(f"Unknown model: {name}")


MODEL_CHOICES = (
    "trivial:every-page",
    "trivial:never-split",
    "text-similarity",
    "tfidf-xgb",
    "ensemble:text-sim+tfidf-xgb",
    "viterbi:text-similarity",
    "viterbi:tfidf-xgb",
    "viterbi:ensemble",
)


def _cmd_eval_run(args: argparse.Namespace) -> int:
    from papercut.data.loaders.hf import HfPssCorpus
    from papercut.eval.runner import evaluate

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}", file=sys.stderr)
        return 2

    corpus = HfPssCorpus.load_from_disk(corpus_path)
    streams = corpus.streams
    cut = max(1, int(len(streams) * args.train_frac))
    train, test = streams[:cut], streams[cut:]
    if not test:
        print("Need at least one test stream; lower --train-frac", file=sys.stderr)
        return 2

    model = _build_model(args.model, corpus)
    if callable(getattr(model, "fit", None)):
        print(f"Fitting {args.model} on {len(train)} streams...")
        model.fit(train)  # type: ignore[attr-defined]

    report = evaluate(model, test)
    print(
        f"{args.model:24s}  page_f1={report.page_f1_mean:.3f}  "
        f"pq={report.pq_mean:.3f}  stp={report.stp:.3f}  "
        f"mndd_mean={report.mndd_mean:.2f}  "
        f"(test_n={report.n_streams})"
    )
    return 0


def _cmd_eval_prospective(args: argparse.Namespace) -> int:
    from papercut.data.loaders.hf import HfPssCorpus
    from papercut.eval.prospective import Slice, format_results, walk_forward

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}", file=sys.stderr)
        return 2

    corpus = HfPssCorpus.load_from_disk(corpus_path)
    n = args.slices
    if n < 2:
        print("--slices must be at least 2", file=sys.stderr)
        return 2
    if len(corpus.streams) < n:
        print(f"Corpus has {len(corpus.streams)} streams but --slices={n}", file=sys.stderr)
        return 2

    chunk = len(corpus.streams) // n
    slices = [
        Slice(name=f"s{i}", streams=corpus.streams[i * chunk : (i + 1) * chunk]) for i in range(n)
    ]
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    models = [_build_model(name, corpus) for name in requested]
    results = walk_forward(models, slices)
    print(format_results(results))
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
    eval_prospective = eval_sub.add_parser(
        "prospective", help="Walk-forward evaluation over consecutive corpus slices."
    )
    eval_prospective.add_argument("--corpus", required=True, help="Saved HfPssCorpus path.")
    eval_prospective.add_argument(
        "--models",
        required=True,
        help=f"Comma-separated model names. Choices: {', '.join(MODEL_CHOICES)}.",
    )
    eval_prospective.add_argument("--slices", type=int, default=4)
    eval_prospective.set_defaults(func=_cmd_eval_prospective)
    eval_run = eval_sub.add_parser("run", help="Evaluate a model on a saved HfPssCorpus pickle.")
    eval_run.add_argument("--corpus", required=True, help="Path to a saved HfPssCorpus.")
    eval_run.add_argument("--model", required=True, choices=MODEL_CHOICES)
    eval_run.add_argument("--train-frac", type=float, default=0.8)
    eval_run.set_defaults(func=_cmd_eval_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
