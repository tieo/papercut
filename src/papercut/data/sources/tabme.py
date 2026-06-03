from __future__ import annotations

from papercut.data.manifest import Labeling, Modality, Source, register

TABME_PP = register(
    Source(
        name="tabme_pp",
        languages=["en"],
        modality=Modality.IMAGE_TEXT,
        labeling=Labeling.PSS_LABELED,
        pages_estimate=3_300_000,
        license="MIT",
        url="https://huggingface.co/datasets/rootsautomation/TABMEpp",
        notes=(
            "Synthetic streams from the Truth Tobacco Industry Documents archive. "
            "Microsoft OCR (better than the original Tesseract-based TABME). "
            "Stored as a per-page bank with stream definitions in streams/<split>_folders.txt. "
            "Use papercut.data.loaders.tabme_pp.load(split, max_streams) to materialize."
        ),
    )
)
