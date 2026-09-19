from __future__ import annotations

import numpy as np

from papercut.models.baselines.tfidf_xgb_layout import _stream_context_features


def test_shape_is_five_blocks_of_the_input() -> None:
    cross = np.arange(12, dtype=np.float32).reshape(4, 3)
    out = _stream_context_features(cross)
    assert out.shape == (4, 15)


def test_edges_repeat_themselves_so_differences_vanish() -> None:
    cross = np.array([[1.0], [2.0], [5.0]], dtype=np.float32)
    out = _stream_context_features(cross)
    previous, following = out[:, 0], out[:, 1]
    assert previous[0] == 1.0
    assert following[-1] == 5.0
    assert out[0, 2] == 0.0
    assert out[-1, 3] == 0.0


def test_neighbour_differences_carry_the_dip() -> None:
    cross = np.array([[0.9], [0.1], [0.9]], dtype=np.float32)
    out = _stream_context_features(cross)
    to_previous, to_following = out[:, 2], out[:, 3]
    assert to_previous[1] < 0
    assert to_following[1] < 0
    assert to_previous[2] > 0


def test_deviation_is_zero_for_a_flat_stream() -> None:
    cross = np.full((5, 2), 0.42, dtype=np.float32)
    out = _stream_context_features(cross)
    assert np.allclose(out[:, -2:], 0.0)


def test_single_pair_stream_is_handled() -> None:
    out = _stream_context_features(np.array([[0.3, 0.7]], dtype=np.float32))
    assert out.shape == (1, 10)
    assert np.allclose(out[:, 4:8], 0.0)
