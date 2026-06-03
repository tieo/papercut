from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from papercut.models.base import ProbabilisticModel
from papercut.streams.types import Stream

_EPS = 1e-9


@dataclass
class SequenceSmoothed:
    """2-state HMM Viterbi smoothing over a probabilistic submodel.

    States: 0 = continuation, 1 = boundary. Transitions, prior, and
    submodel are calibrated/fit on training data; emission uses the
    submodel's per-page boundary probability re-normalised by the
    boundary rate so it functions as a likelihood ratio.
    """

    submodel: ProbabilisticModel
    name: str = field(default="viterbi_smoothed")
    boundary_rate: float = 0.05
    same_to_new: float = 0.05
    new_to_new: float = 0.05

    def fit(self, streams: list[Stream]) -> None:
        n_boundaries = 0
        n_positions = 0
        n_same_to_new = 0
        n_same_to_same = 0
        n_new_to_new = 0
        n_new_to_same = 0
        for stream in streams:
            if stream.boundaries is None:
                continue
            b = stream.boundaries
            for i in range(1, len(b)):
                n_positions += 1
                if b[i]:
                    n_boundaries += 1
                prev_new = b[i - 1]
                curr_new = b[i]
                if prev_new and curr_new:
                    n_new_to_new += 1
                elif prev_new:
                    n_new_to_same += 1
                elif curr_new:
                    n_same_to_new += 1
                else:
                    n_same_to_same += 1
        if n_positions:
            self.boundary_rate = n_boundaries / n_positions
        same_total = n_same_to_same + n_same_to_new
        new_total = n_new_to_same + n_new_to_new
        if same_total:
            self.same_to_new = max(_EPS, min(1 - _EPS, n_same_to_new / same_total))
        if new_total:
            self.new_to_new = max(_EPS, min(1 - _EPS, n_new_to_new / new_total))
        if callable(getattr(self.submodel, "fit", None)):
            self.submodel.fit(streams)  # type: ignore[attr-defined]

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        return self.submodel.predict_probs(stream)

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        return self._viterbi(self.submodel.predict_probs(stream))

    def _log_emission(self, p: float, state: int) -> float:
        """Treat submodel posterior as emission likelihood.

        Using `log p` for state=1 and `log (1 - p)` for state=0 means the
        transition prior (same_to_new, new_to_new) provides the smoothing.
        An earlier Bayesian inversion that divided by the boundary rate
        massively over-weighted the boundary state on imbalanced corpora.
        """
        if state == 1:
            return math.log(max(p, _EPS))
        return math.log(max(1 - p, _EPS))

    def _viterbi(self, probs: Sequence[float]) -> tuple[bool, ...]:
        n = len(probs)
        if n == 1:
            return (True,)

        # log transition matrix: log_t[prev][curr]
        log_t = [
            [
                math.log(max(_EPS, 1 - self.same_to_new)),
                math.log(self.same_to_new),
            ],
            [
                math.log(max(_EPS, 1 - self.new_to_new)),
                math.log(self.new_to_new),
            ],
        ]

        delta_prev = (-math.inf, 0.0)
        back: list[tuple[int, int]] = [(0, 0)]
        for t in range(1, n):
            em0 = self._log_emission(probs[t], 0)
            em1 = self._log_emission(probs[t], 1)
            c00 = delta_prev[0] + log_t[0][0] + em0
            c10 = delta_prev[1] + log_t[1][0] + em0
            c01 = delta_prev[0] + log_t[0][1] + em1
            c11 = delta_prev[1] + log_t[1][1] + em1
            d0, b0 = (c00, 0) if c00 >= c10 else (c10, 1)
            d1, b1 = (c01, 0) if c01 >= c11 else (c11, 1)
            delta_prev = (d0, d1)
            back.append((b0, b1))

        states = [0] * n
        states[-1] = 0 if delta_prev[0] > delta_prev[1] else 1
        for t in range(n - 1, 0, -1):
            states[t - 1] = back[t][states[t]]
        states[0] = 1
        return tuple(s == 1 for s in states)
