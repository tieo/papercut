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


def _cmd_data_filter(args: argparse.Namespace) -> int:
    """Filter a saved corpus to streams matching length constraints."""
    from papercut.data.loaders.hf import HfPssCorpus

    src = Path(args.in_path)
    if not src.exists():
        print(f"Corpus not found: {src}", file=sys.stderr)
        return 2
    corpus = HfPssCorpus.load_from_disk(src)
    kept_streams = [s for s in corpus.streams if args.min_pages <= len(s) <= args.max_pages]
    if not kept_streams:
        print("No streams matched the filter", file=sys.stderr)
        return 2
    kept_pages: set[object] = {p for s in kept_streams for p in s.pages}
    kept_texts = {p: t for p, t in corpus._texts.items() if p in kept_pages}
    out_corpus = HfPssCorpus(streams=kept_streams, _texts=kept_texts)
    out_corpus.save(Path(args.out))
    print(
        f"Kept {len(kept_streams)} / {len(corpus.streams)} streams "
        f"({sum(len(s) for s in kept_streams)} pages) -> {args.out}"
    )
    return 0


def _cmd_data_resample(args: argparse.Namespace) -> int:
    """Resample new short streams from the unique docs in a corpus.

    Each "doc" is the set of pages sharing the same `PageRef.source` (since
    our HF adapter encodes `<namespace>/<doc_id>` there). We sample a Poisson
    number of distinct docs per new stream and concatenate them, yielding
    shorter streams more representative of personal scan-stacks.
    """
    import math

    from papercut.data.loaders.hf import HfPssCorpus
    from papercut.streams.types import PageRef, Stream

    src = Path(args.in_path)
    if not src.exists():
        print(f"Corpus not found: {src}", file=sys.stderr)
        return 2
    corpus = HfPssCorpus.load_from_disk(src)

    docs: dict[str, list[PageRef]] = {}
    for stream in corpus.streams:
        for page in stream.pages:
            docs.setdefault(page.source, []).append(page)
    unique_docs = sorted(docs.keys())
    if not unique_docs:
        print("Source corpus has no docs", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)

    def _poisson(lam: float) -> int:
        target = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= rng.random()
            if p < target:
                return k - 1

    new_streams: list[Stream] = []
    for _ in range(args.n_streams):
        n_docs = max(1, min(len(unique_docs), _poisson(args.mean_docs)))
        chosen = rng.sample(unique_docs, k=n_docs)
        pages: list[PageRef] = []
        boundaries: list[bool] = []
        for doc_id in chosen:
            doc_pages = sorted(docs[doc_id], key=lambda p: p.page)
            for i, p in enumerate(doc_pages):
                pages.append(p)
                boundaries.append(i == 0)
        new_streams.append(Stream(pages=tuple(pages), boundaries=tuple(boundaries)))

    out = HfPssCorpus(streams=new_streams, _texts=dict(corpus._texts))
    out.save(Path(args.out))
    print(
        f"Resampled {len(new_streams)} streams "
        f"(mean_docs={args.mean_docs}, "
        f"{sum(len(s) for s in new_streams)} pages) -> {args.out}"
    )
    return 0


def _cmd_data_fair_split(args: argparse.Namespace) -> int:
    """Resample disjoint-doc train/test streams from a corpus.

    Unique docs (PageRef.source values) are partitioned by `train_frac`.
    Train streams sample from the train docs only, test streams from the
    test docs only. Eliminates the memorisation confound from the simpler
    resample-streams command.
    """
    import math

    from papercut.data.loaders.hf import HfPssCorpus
    from papercut.streams.types import PageRef, Stream

    src = Path(args.in_path)
    if not src.exists():
        print(f"Corpus not found: {src}", file=sys.stderr)
        return 2
    corpus = HfPssCorpus.load_from_disk(src)

    docs: dict[str, list[PageRef]] = {}
    for stream in corpus.streams:
        for page in stream.pages:
            docs.setdefault(page.source, []).append(page)
    unique_docs = sorted(docs.keys())
    rng = random.Random(args.seed)
    rng.shuffle(unique_docs)
    cut = max(1, int(len(unique_docs) * args.train_frac))
    train_docs, test_docs = unique_docs[:cut], unique_docs[cut:]
    if not train_docs or not test_docs:
        print("Both splits must be non-empty", file=sys.stderr)
        return 2

    def _poisson(lam: float) -> int:
        target = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= rng.random()
            if p < target:
                return k - 1

    def _build_streams(doc_pool: list[str], n_streams: int) -> list[Stream]:
        streams: list[Stream] = []
        for _ in range(n_streams):
            n_docs = max(1, min(len(doc_pool), _poisson(args.mean_docs)))
            chosen = rng.sample(doc_pool, k=n_docs)
            pages: list[PageRef] = []
            boundaries: list[bool] = []
            for doc_id in chosen:
                doc_pages = sorted(docs[doc_id], key=lambda p: p.page)
                for i, p in enumerate(doc_pages):
                    pages.append(p)
                    boundaries.append(i == 0)
            streams.append(Stream(pages=tuple(pages), boundaries=tuple(boundaries)))
        return streams

    train_streams = _build_streams(train_docs, args.n_train)
    test_streams = _build_streams(test_docs, args.n_test)

    texts = dict(corpus._texts)
    HfPssCorpus(streams=train_streams, _texts=texts).save(Path(args.train_out))
    HfPssCorpus(streams=test_streams, _texts=texts).save(Path(args.test_out))
    print(
        f"Train: {len(train_streams)} streams "
        f"({sum(len(s) for s in train_streams)} pages, {len(train_docs)} unique docs) -> {args.train_out}"
    )
    print(
        f"Test:  {len(test_streams)} streams "
        f"({sum(len(s) for s in test_streams)} pages, {len(test_docs)} unique docs) -> {args.test_out}"
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
    if name == "tfidf-xgb-rich":
        from papercut.models.baselines.tfidf_xgb_rich import TfIdfXgbRich

        return TfIdfXgbRich(resolver=resolver)  # type: ignore[arg-type]
    if name == "ensemble:text-sim+tfidf-xgb-rich":
        from papercut.models.baselines.text_similarity import TextSimilarityBaseline
        from papercut.models.baselines.tfidf_xgb_rich import TfIdfXgbRich
        from papercut.models.ensembles.late import LateEnsemble

        return LateEnsemble(
            submodels=[
                TextSimilarityBaseline(resolver=resolver),  # type: ignore[arg-type]
                TfIdfXgbRich(resolver=resolver),  # type: ignore[arg-type]
            ],
            name="ensemble:text-sim+tfidf-xgb-rich",
        )
    if name == "multilingual-minilm":
        from papercut.models.baselines.multilingual_minilm import MultilingualMiniLM

        return MultilingualMiniLM(resolver=resolver)  # type: ignore[arg-type]
    if name == "ensemble:rich+minilm":
        from papercut.models.baselines.multilingual_minilm import MultilingualMiniLM
        from papercut.models.baselines.tfidf_xgb_rich import TfIdfXgbRich
        from papercut.models.ensembles.late import LateEnsemble

        return LateEnsemble(
            submodels=[
                TfIdfXgbRich(resolver=resolver),  # type: ignore[arg-type]
                MultilingualMiniLM(resolver=resolver),  # type: ignore[arg-type]
            ],
            name="ensemble:rich+minilm",
        )
    if name == "minilm-xgb":
        from papercut.models.baselines.minilm_xgb import MiniLMXgb

        return MiniLMXgb(resolver=resolver)  # type: ignore[arg-type]
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
    "tfidf-xgb-rich",
    "ensemble:text-sim+tfidf-xgb",
    "ensemble:text-sim+tfidf-xgb-rich",
    "multilingual-minilm",
    "minilm-xgb",
    "ensemble:rich+minilm",
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
    if args.test_corpus:
        test_path = Path(args.test_corpus)
        if not test_path.exists():
            print(f"Test corpus not found: {test_path}", file=sys.stderr)
            return 2
        test_corpus = HfPssCorpus.load_from_disk(test_path)
        train, test = corpus.streams, test_corpus.streams
        combined_texts = {**corpus._texts, **test_corpus._texts}
        resolver_corpus = HfPssCorpus(
            streams=[*corpus.streams, *test_corpus.streams], _texts=combined_texts
        )
    else:
        streams = corpus.streams
        cut = max(1, int(len(streams) * args.train_frac))
        train, test = streams[:cut], streams[cut:]
        if not test:
            print("Need at least one test stream; lower --train-frac", file=sys.stderr)
            return 2
        resolver_corpus = corpus

    model = _build_model(args.model, resolver_corpus)
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

    filt = data_sub.add_parser("filter", help="Filter a saved corpus by per-stream page count.")
    filt.add_argument("--in", dest="in_path", required=True, help="Source corpus path.")
    filt.add_argument("--out", required=True, help="Filtered corpus path.")
    filt.add_argument("--max-pages", type=int, default=10**9)
    filt.add_argument("--min-pages", type=int, default=1)
    filt.set_defaults(func=_cmd_data_filter)

    resamp = data_sub.add_parser(
        "resample-streams",
        help="Resample new streams from the unique docs in a corpus.",
    )
    resamp.add_argument("--in", dest="in_path", required=True)
    resamp.add_argument("--out", required=True)
    resamp.add_argument("--n-streams", type=int, default=200)
    resamp.add_argument("--mean-docs", type=float, default=3.0)
    resamp.add_argument("--seed", type=int, default=0)
    resamp.set_defaults(func=_cmd_data_resample)

    fair = data_sub.add_parser(
        "fair-split",
        help="Resample disjoint-doc train/test streams.",
    )
    fair.add_argument("--in", dest="in_path", required=True)
    fair.add_argument("--train-out", required=True)
    fair.add_argument("--test-out", required=True)
    fair.add_argument("--n-train", type=int, default=400)
    fair.add_argument("--n-test", type=int, default=100)
    fair.add_argument("--mean-docs", type=float, default=3.0)
    fair.add_argument("--train-frac", type=float, default=0.8)
    fair.add_argument("--seed", type=int, default=0)
    fair.set_defaults(func=_cmd_data_fair_split)

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
    eval_run.add_argument(
        "--test-corpus",
        default=None,
        help="Optional separate test corpus path; overrides --train-frac.",
    )
    eval_run.set_defaults(func=_cmd_eval_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
