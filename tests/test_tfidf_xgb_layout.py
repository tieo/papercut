from __future__ import annotations

from papercut.data.loaders.hf import HfPssCorpus
from papercut.models.baselines.tfidf_xgb_layout import TfIdfXgbLayout


def test_layout_model_passes_column_sampling_to_xgboost() -> None:
    model = TfIdfXgbLayout(corpus=HfPssCorpus(streams=[]), colsample_bytree=0.5)
    assert model.model.get_params()["colsample_bytree"] == 0.5
