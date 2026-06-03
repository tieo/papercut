from __future__ import annotations

from pathlib import Path

from papercut.data.manifest import Labeling, Modality, Source, register

TABME_PP = register(
    Source(
        name="tabme_pp",
        languages=["en"],
        modality=Modality.IMAGE_TEXT,
        labeling=Labeling.PSS_LABELED,
        pages_estimate=3_300_000,
        license="MIT",
        url="https://huggingface.co/datasets/rootsautomation/TABMEpp",
        notes=(
            "Synthetic streams from the Truth Tobacco Industry Documents archive. "
            "Microsoft OCR (better than the original Tesseract-based TABME). "
            "~110k streams, 1.2M docs (44.8k unique), 3.3M pages."
        ),
    )
)


def download(target_dir: Path, max_streams: int | None = None) -> Path:
    """Pull TABME++ from HuggingFace Hub into target_dir/tabme_pp/.

    Requires the 'ml' optional extra. Pass `max_streams` to dev on a tiny
    slice (e.g. 100) before running a full download.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("TABME++ download requires the 'ml' extra: `uv sync --extra ml`") from e

    dest = target_dir / TABME_PP.name
    dest.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("rootsautomation/TABMEpp", split="train")
    if max_streams is not None:
        ds = ds.select(range(min(max_streams, len(ds))))
    ds.save_to_disk(str(dest))
    return dest
