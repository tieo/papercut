from __future__ import annotations

from papercut.data.manifest import Labeling, Modality, Source, register

BUNDESTAG_DRUCKSACHEN = register(
    Source(
        name="bundestag_drucksachen",
        languages=["de"],
        modality=Modality.IMAGE_TEXT,
        labeling=Labeling.SINGLE_DOC_PDFS,
        pages_estimate=None,
        license="public (Bundestag Open Data)",
        url="https://www.bundestag.de/services/opendata",
        notes=(
            "German parliamentary documents (Drucksachen) since 1949 plus plenary "
            "protocols. Cleanest German source for synthetic PSS streams. "
            "~12k Drucksachen per 4-year period, ranging 1 to 3000+ pages."
        ),
    )
)
