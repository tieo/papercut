from __future__ import annotations

from papercut.data.manifest import Labeling, Modality, Source, register

TOBACCO800 = register(
    Source(
        name="tobacco800",
        languages=["en"],
        modality=Modality.IMAGE_TEXT,
        labeling=Labeling.PSS_LABELED,
        pages_estimate=1_290,
        license="public domain (US legal proceedings)",
        url="https://tc11.cvc.uab.es/datasets/Tobacco800_1",
        notes=(
            "Classic small benchmark. 67% singletons inflates degenerate baselines "
            "(every-page-new-doc reaches F1 0.77). Useful for literature comparison only."
        ),
    )
)
