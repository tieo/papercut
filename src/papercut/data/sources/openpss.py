from __future__ import annotations

from papercut.data.manifest import Labeling, Modality, Source, register

OPENPSS_LONG = register(
    Source(
        name="openpss_long",
        languages=["nl"],
        modality=Modality.IMAGE_TEXT,
        labeling=Labeling.PSS_LABELED,
        pages_estimate=89_491,
        license="public (FOIA-released)",
        url="https://irlab.science.uva.nl/",
        notes=(
            "Dutch FOIA from COVID-era ministry requests, manually annotated. "
            "110 streams, 24,181 docs, median 217 pages per stream."
        ),
    )
)

OPENPSS_SHORT = register(
    Source(
        name="openpss_short",
        languages=["nl"],
        modality=Modality.IMAGE_TEXT,
        labeling=Labeling.PSS_LABELED,
        pages_estimate=52_177,
        license="public (FOIA-released)",
        url="https://irlab.science.uva.nl/",
        notes=(
            "Dutch FOIA constructed from zip archives (true boundaries known). "
            "312 streams, 8,162 docs, median 60 pages per stream."
        ),
    )
)
