from __future__ import annotations

import papercut.data.sources  # noqa: F401
from papercut.data.manifest import REGISTRY


def test_tabme_pp_registered() -> None:
    assert "tabme_pp" in REGISTRY
    src = REGISTRY["tabme_pp"]
    assert src.languages == ["en"]
    assert src.url.startswith("https://huggingface.co/")
    assert src.pages_estimate is not None and src.pages_estimate > 1_000_000


def test_register_rejects_duplicates() -> None:
    import pytest

    from papercut.data.manifest import Labeling, Modality, Source, register

    dup = Source(
        name="tabme_pp",
        languages=["en"],
        modality=Modality.IMAGE_TEXT,
        labeling=Labeling.PSS_LABELED,
        url="https://example.com",
    )
    with pytest.raises(ValueError, match="already registered"):
        register(dup)
