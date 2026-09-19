from __future__ import annotations

import numpy as np

from papercut.models.baselines.tfidf_xgb_layout import standardise_within_stream


def test_columns_become_zero_mean_and_unit_spread() -> None:
    block = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)
    out = standardise_within_stream(block)
    assert abs(float(out.mean())) < 1e-5
    assert abs(float(out.std()) - 1.0) < 1e-3


def test_two_streams_of_different_scale_map_to_the_same_standing() -> None:
    dense = np.array([[0.40], [0.44], [0.48]], dtype=np.float32)
    sparse = np.array([[0.04], [0.08], [0.12]], dtype=np.float32)
    assert np.allclose(standardise_within_stream(dense), standardise_within_stream(sparse), atol=1e-4)


def test_a_flat_stream_does_not_explode() -> None:
    out = standardise_within_stream(np.full((4, 2), 0.3, dtype=np.float32))
    assert np.all(np.abs(out) < 1.0)


def test_a_single_pair_has_no_standing() -> None:
    out = standardise_within_stream(np.array([[0.7, 0.2]], dtype=np.float32))
    assert out.shape == (1, 2)
    assert np.allclose(out, 0.0)
