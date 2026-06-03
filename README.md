# papercut

[![ci](https://github.com/tieo/papercut/actions/workflows/ci.yml/badge.svg)](https://github.com/tieo/papercut/actions/workflows/ci.yml)

Page Stream Segmentation for [paperless-ngx](https://github.com/paperless-ngx/paperless-ngx). Scan a stack, walk away. papercut figures out where one document ends and the next begins, so each gets ingested separately.

## Goal

- **≥95% Straight-Through Processing (STP)**: fraction of scan stacks segmented perfectly, no manual cleanup.
- **CPU-only inference**: runs alongside paperless on a normal home server.
- **Language-agnostic**: no hardcoded `"Page X of Y"` regex. Vision and multilingual text encoders learn the universal cues (letterheads, address blocks, signature blocks, page-number patterns) regardless of language.
- **"Reasonably well on anything, perfect on common stuff."** Graceful degradation; low-confidence cuts surfaced for review rather than silent guesses.

## Status

Pre-alpha. The evaluation harness, three baselines (two trivial plus a trainable TF-IDF + XGBoost), prospective walk-forward eval, stream construction from PDFs, and a registry of 11 multilingual data sources all work end to end. No real-data model trained yet.

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
├── data/        # dataset registry and per-source downloaders
├── ocr/         # unified OCR interface (planned: PaddleOCR primary, Tesseract fallback)
├── streams/     # PDF concatenation, boundary labeling, PageResolver, scan augmentation
├── models/      # one module per architecture, all implementing models.base.Model
├── eval/        # metrics (page-F1, doc-F1, Panoptic Quality, STP, MNDD) plus prospective harness
└── serve/       # paperless-ngx integration (planned: consume-folder watcher and pre-consume hook)
```

## Dev quickstart

```bash
uv sync --dev
uv run pytest
uv run papercut sources list
uv run papercut eval baseline-smoke
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
papercut streams build <pdf-dir> <out-dir> [--n-streams 100] [--mean-docs 10]
papercut eval baseline-smoke
papercut eval prospective-smoke
```

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
