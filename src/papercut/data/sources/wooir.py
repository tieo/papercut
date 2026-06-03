from __future__ import annotations

from papercut.data.manifest import Labeling, Modality, Source, register

WOOIR = register(
    Source(
        name="wooir",
        languages=["nl"],
        modality=Modality.IMAGE_TEXT,
        labeling=Labeling.PSS_LABELED,
        pages_estimate=44_975,
        license="public (FOIA-released)",
        url="https://irlab.science.uva.nl/resources/wooir_pss",
        notes=(
            "Dutch FOIA release, predecessor to OpenPSS. 229 streams, 7,118 docs, "
            "32% singletons. Two corpora with separate train/test splits."
        ),
    )
)
