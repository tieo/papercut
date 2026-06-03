from __future__ import annotations

from papercut.data.manifest import Labeling, Modality, Source, register

EURLEX = register(
    Source(
        name="eurlex",
        languages=[
            "bg",
            "cs",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "fi",
            "fr",
            "ga",
            "hr",
            "hu",
            "it",
            "lt",
            "lv",
            "mt",
            "nl",
            "pl",
            "pt",
            "ro",
            "sk",
            "sl",
            "sv",
        ],
        modality=Modality.IMAGE_TEXT,
        labeling=Labeling.SINGLE_DOC_PDFS,
        pages_estimate=None,
        license="EU public data (free reuse)",
        url="https://eur-lex.europa.eu/content/help/data-reuse/reuse-contents-eurlex-details.html",
        notes=(
            "Official Journals and legal acts across all 24 EU languages since 1950s. "
            "PDF + XML. Best single source for multilingual government-format docs. "
            "Time-stratified for prospective eval (e.g. reserve 2025+ for held-out)."
        ),
    )
)
