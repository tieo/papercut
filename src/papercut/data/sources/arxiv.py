from __future__ import annotations

from papercut.data.manifest import Labeling, Modality, Source, register

ARXIV = register(
    Source(
        name="arxiv",
        languages=["en"],
        modality=Modality.IMAGE_TEXT,
        labeling=Labeling.SINGLE_DOC_PDFS,
        pages_estimate=None,
        license="non-exclusive license to distribute (per paper)",
        url="https://info.arxiv.org/help/bulk_data_s3.html",
        notes=(
            "Bulk PDFs via the requester-pays S3 bucket. Long multi-page documents "
            "with consistent layout per arXiv category. Natural time stratification "
            "for prospective eval (reserve future months)."
        ),
    )
)
