# papercut

[![ci](https://github.com/tieo/papercut/actions/workflows/ci.yml/badge.svg)](https://github.com/tieo/papercut/actions/workflows/ci.yml)

Page Stream Segmentation for [paperless-ngx](https://github.com/paperless-ngx/paperless-ngx). Scan a stack, walk away. papercut figures out where one document ends and the next begins, so each gets ingested separately.

## Goal

- **≥95% Straight-Through Processing (STP)**: fraction of scan stacks segmented perfectly, no manual cleanup.
- **CPU-only inference**: runs alongside paperless on a normal home server.
- **Language-agnostic**: no hardcoded `"Page X of Y"` regex. Vision and multilingual text encoders learn the universal cues (letterheads, address blocks, signature blocks, page-number patterns) regardless of language.
- **"Reasonably well on anything, perfect on common stuff."** Graceful degradation; low-confidence cuts surfaced for review rather than silent guesses.

## Status

Pre-alpha. The pipeline runs end to end on real TABME++ data: download a slice with `papercut data download`, evaluate any registered model with `papercut eval run`. Four baselines so far (two trivial, one zero-shot text-similarity, one trainable TF-IDF + XGBoost). Bigger models (DiT, LayoutXLM, multilingual MiniLM) and OCR-on-scan integration are next.

### Measured baseline numbers

100 TABME++ test streams (3,103 pages). Single train/test split (80/20):

| Model | page F1 | PQ | STP | MNDD |
|---|---|---|---|---|
| trivial:every-page | 0.510 | 0.258 | 0.000 | 19.55 |
| trivial:never-split | 0.000 | 0.014 | 0.000 | 20.35 |
| **text-similarity** | **0.688** | 0.492 | **0.050** | 10.40 |
| tfidf-xgb | 0.612 | **0.485** | 0.000 | **9.90** |

Walk-forward prospective eval, 4 consecutive slices (train on s0..s_{i-1}, test on s_i):

| Model | slice | page F1 | PQ | MNDD |
|---|---|---|---|---|
| text-similarity | s1 / s2 / s3 | 0.683 / 0.638 / 0.700 | 0.491 / 0.454 / 0.500 | 12.60 / 14.48 / 10.12 |
| tfidf-xgb       | s1 / s2 / s3 | 0.644 / 0.608 / 0.633 | 0.513 / 0.472 / 0.490 |  10.16 /  10.08 /  9.32 |

text-similarity (char-4-gram Jaccard between consecutive pages, zero training, language-agnostic) consistently wins on page F1. tfidf-xgb consistently wins on document-level metrics (PQ, MNDD), suggesting the two are complementary and an ensemble is the obvious next step. STP stays near zero on this corpus because TABME++ streams are long (median 30 pages); the target use case (personal mail, 5 to 15 docs per stack) should fare much better.

## Approach

The field is **Page Stream Segmentation (PSS)**. The current SOTA on the public TABME++ benchmark is around 80% STP using a fine-tuned 7B LLM, which is impressive but not CPU-friendly. For our constraints we follow the cheaper, language-agnostic direction:

- Per-page representation: vision embedding (DiT or a small CNN) concatenated with a multilingual text embedding (MiniLM, mE5).
- Boundary classifier over consecutive pages.
- Sequence smoothing (BiLSTM or Viterbi) over per-page boundary probabilities so global signals like recurring footers can override noisy local decisions.

Evaluation is **prospective**: train on a frozen snapshot, evaluate on data collected after that snapshot. Mirrors how the system will be used in practice and prevents overfitting to a static test split.

## Layout

```
src/papercut/
├── cli/         # `papercut` command-line entry point
├── data/        # dataset registry, per-source downloaders, HF adapter
├── ocr/         # unified OCR interface (planned: PaddleOCR primary, Tesseract fallback)
├── streams/     # PDF concatenation, boundary labeling, PageResolver, scan augmentation
├── models/      # one module per architecture, all implementing models.base.Model
├── eval/        # metrics (page-F1, doc-F1, Panoptic Quality, STP, MNDD) plus prospective harness
└── serve/       # paperless-ngx integration (planned: consume-folder watcher and pre-consume hook)
```

## Quickstart

```bash
uv sync --dev
uv run pytest

# Pull a tiny TABME++ slice and run all baselines on it.
uv run papercut data download tabme-pp --split test --max-streams 5 --out data/tabme_pp_5.pkl
uv run papercut eval run --corpus data/tabme_pp_5.pkl --model text-similarity
uv run papercut eval run --corpus data/tabme_pp_5.pkl --model tfidf-xgb
```

On **NixOS**, wrap calls in `nix-shell` so binary wheels (numpy, scipy, sklearn, xgboost, future torch) can load:

```bash
nix-shell --run "uv sync --dev"
nix-shell --run "uv run pytest"
```

Data lives in `./data/` (gitignored).

## CLI

```
papercut sources list
papercut data download tabme-pp [--split SPLIT] [--max-streams N] [--out PATH]
papercut streams build <pdf-dir> <out-dir> [--n-streams 100] [--mean-docs 10]
papercut eval run --corpus PATH --model MODEL [--train-frac 0.8]
papercut eval baseline-smoke
papercut eval prospective-smoke
```

Models registered for `papercut eval run --model`:
`trivial:every-page`, `trivial:never-split`, `text-similarity`, `tfidf-xgb`.

## Registered sources

| Source | Pages | Languages | Labeling |
|---|---|---|---|
| tabme_pp | 3.3M | en | PSS-labeled |
| iit_cdip | 7M | en | bulk |
| openpss_long | 89k | nl | PSS-labeled |
| openpss_short | 52k | nl | PSS-labeled |
| wooir | 45k | nl | PSS-labeled |
| ai_lab_splitter | 32k | pt | PSS-labeled (image-only) |
| doclaynet | 80k | en, de, fr, ja | single-doc PDFs |
| tobacco800 | 1.3k | en | PSS-labeled |
| eurlex | (bulk) | 24 EU languages | single-doc PDFs |
| bundestag_drucksachen | (bulk) | de | single-doc PDFs |
| arxiv | (bulk) | en | single-doc PDFs |

## License

Apache-2.0.
