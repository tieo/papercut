from __future__ import annotations

from papercut.data.manifest import Labeling, Modality, Source, register

DOCLAYNET = register(
    Source(
        name="doclaynet",
        languages=["en", "de", "fr", "ja"],
        modality=Modality.IMAGE_TEXT,
        labeling=Labeling.SINGLE_DOC_PDFS,
        pages_estimate=80_863,
        license="CDLA-Permissive-1.0",
        url="https://huggingface.co/datasets/docling-project/DocLayNet",
        notes=(
            "Heterogeneous document layout corpus (magazines, government, finance, "
            "patents, manuals, scientific). Multilingual. Concatenate for synthetic "
            "PSS streams with strong layout variability."
        ),
    )
)
