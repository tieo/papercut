from __future__ import annotations

from papercut.data.manifest import Labeling, Modality, Source, register

IIT_CDIP = register(
    Source(
        name="iit_cdip",
        languages=["en"],
        modality=Modality.IMAGE_TEXT,
        labeling=Labeling.BULK_CORPUS,
        pages_estimate=7_000_000,
        license="public domain (US legal proceedings)",
        url="https://data.nist.gov/od/id/mds2-2531",
        notes=(
            "~7 million scanned documents from the Truth Tobacco Industry archive. "
            "Parent of Tobacco800, RVL-CDIP, TABME. Multi-TB. 90s-era OCR included; "
            "consider re-OCRing for quality."
        ),
    )
)
