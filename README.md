# papercut

Page Stream Segmentation for [paperless-ngx](https://github.com/paperless-ngx/paperless-ngx). Scan a stack, walk away. papercut figures out where one document ends and the next begins, so each gets ingested separately.

## Goal

- **≥95% Straight-Through Processing (STP)**: fraction of scan stacks segmented perfectly, no manual cleanup.
- **CPU-only inference**: runs alongside paperless on a normal home server.
- **Language-agnostic**: no hardcoded `"Page X of Y"` regex. Vision and multilingual text encoders learn the universal cues (letterheads, address blocks, signature blocks, page-number patterns) regardless of language.
- **"Reasonably well on anything, perfect on common stuff."** Graceful degradation; low-confidence cuts surfaced for review rather than silent guesses.

## Status

Pre-alpha. Repo scaffolding and core interfaces. No working model yet.

## Approach

This is the field known as **Page Stream Segmentation (PSS)** in the literature. The published SOTA on the public TABME++ benchmark is around 80% STP using a fine-tuned 7B LLM, which is impressive but not CPU-friendly. For our constraints we follow the cheaper, language-agnostic direction:

- Per-page representation: vision embedding (DiT or a small CNN) concatenated with a multilingual text embedding (MiniLM, mE5).
- Boundary classifier over consecutive pages.
- Sequence smoothing (BiLSTM or Viterbi) over per-page boundary probabilities so global signals like recurring footers can override noisy local decisions.

Evaluation is **prospective**: train on a frozen snapshot, evaluate on data collected after that snapshot. Mirrors how the system will be used in practice and prevents overfitting to a static test split.

## Layout

```
src/papercut/
├── data/        # dataset registry and per-source downloaders
├── ocr/         # unified OCR interface (PaddleOCR primary, Tesseract fallback)
├── streams/     # PDF concatenation, boundary labeling, scan augmentation
├── models/      # one module per architecture, all implementing models.base.Model
├── eval/        # metrics (page-F1, doc-F1, Panoptic Quality, STP, MNDD) and prospective harness
└── serve/       # paperless-ngx integration (consume-folder watcher and pre-consume hook)
```

## Dev quickstart

```bash
uv sync --dev
uv run pytest
uv run ruff check
```

Data lives in `./data/` (gitignored). Configurable via env var when that becomes painful.

## License

Apache-2.0.
