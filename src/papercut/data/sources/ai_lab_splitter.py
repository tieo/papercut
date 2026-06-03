from __future__ import annotations

from papercut.data.manifest import Labeling, Modality, Source, register

AI_LAB_SPLITTER = register(
    Source(
        name="ai_lab_splitter",
        languages=["pt"],
        modality=Modality.IMAGE,
        labeling=Labeling.PSS_LABELED,
        pages_estimate=31_789,
        license="research use",
        url="https://www.sciencedirect.com/science/article/abs/pii/S0952197621002426",
        notes=(
            "Brazilian court proceedings, 1,869 streams. Image-only at 224x224, "
            "so text cannot be OCR'd from this dataset; useful for vision-only "
            "evaluation and as a stress test for OOD layout."
        ),
    )
)
