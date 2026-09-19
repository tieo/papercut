"""Decide boundaries by their standing within a stream, not by an absolute cut.

A model trained on one distribution and run on another keeps much of its
ordering and loses its scale: on a real scanner stack the probabilities pile
up at one end, so a fixed threshold either splits every page or none, while
the ranking still puts true boundaries above false ones. These decoders read
the ranking and ignore the scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from papercut.models.base import ProbabilisticModel
from papercut.streams.types import Stream


def rank_quantiles(probs: Sequence[float]) -> tuple[float, ...]:
    """Map scores to their position within this stream, ties averaged."""
    if not probs:
        return ()
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    quantiles = [0.0] * len(probs)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and probs[order[stop + 1]] == probs[order[index]]:
            stop += 1
        share = (index + stop) / 2 / max(1, len(probs) - 1)
        for position in range(index, stop + 1):
            quantiles[order[position]] = share
        index = stop + 1
    return tuple(quantiles)


@dataclass
class RankNormalized:
    """Replace a submodel's probabilities with their rank inside the stream.

    Wrapping restores a usable threshold when a submodel's scores saturate,
    since a quantile of 0.9 means the same thing in every stream whatever the
    raw numbers do.
    """

    submodel: ProbabilisticModel
    threshold: float = 0.75
    name: str = field(default="rank_normalized")

    def fit(self, streams: Sequence[Stream]) -> None:
        if callable(getattr(self.submodel, "fit", None)):
            self.submodel.fit(streams)  # type: ignore[attr-defined]

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        probs = self.submodel.predict_probs(stream)
        if len(probs) < 2:
            return probs
        return (1.0, *rank_quantiles(probs[1:]))

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        probs = self.predict_probs(stream)
        return (True, *(p >= self.threshold for p in probs[1:]))


@dataclass
class ExpectedCount:
    """Open as many documents as the stream's length suggests, highest scores first.

    Document length is the one property of a stack that carries over from the
    training corpus even when the scores do not: mail is mail, whoever sent
    it. A stream of twenty pages at five pages per document opens four, so the
    decoder takes the four best-scoring pages rather than everything above a
    number that no longer means anything.
    """

    submodel: ProbabilisticModel
    pages_per_document: float = 5.0
    name: str = field(default="expected_count")

    def fit(self, streams: Sequence[Stream]) -> None:
        labeled = [s for s in streams if s.boundaries is not None]
        pages = sum(len(s.pages) for s in labeled)
        documents = sum(sum(s.boundaries) for s in labeled)
        if documents:
            self.pages_per_document = pages / documents
        if callable(getattr(self.submodel, "fit", None)):
            self.submodel.fit(labeled)  # type: ignore[attr-defined]

    def fit_length(self, streams: Sequence[Stream]) -> None:
        """Take only the length prior from labeled streams, leaving the submodel."""
        labeled = [s for s in streams if s.boundaries is not None]
        pages = sum(len(s.pages) for s in labeled)
        documents = sum(sum(s.boundaries) for s in labeled)
        if documents:
            self.pages_per_document = pages / documents

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        return self.submodel.predict_probs(stream)

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        probs = self.predict_probs(stream)
        pages = len(probs)
        if pages < 2:
            return (True,) * pages
        wanted = max(1, round(pages / max(1.0, self.pages_per_document)))
        extra = min(pages - 1, wanted - 1)
        chosen = sorted(range(1, pages), key=lambda i: -probs[i])[:extra]
        picked = set(chosen)
        return (True, *(i in picked for i in range(1, pages)))


@dataclass
class AdaptiveDecode:
    """Use the threshold while it discriminates, the ranking once it stops.

    An absolute cut is the better rule on the distribution a model was fitted
    to, and it collapses on another one, where the scores crowd together and
    the cut lands above or below all of them. Ranking is the reverse: robust
    to that crowding, and wasteful on a stream holding one document, since it
    always opens a fixed share of the pages.

    Two symptoms mark the second case. Most pages clearing the cut says the
    scores have drifted upward wholesale, since a stack where four pages in
    five open a document does not exist. Scores packed into a narrow band says
    the model is answering the same thing everywhere and the cut lands on one
    side of all of them. A stream where no page clears a well spread set of
    scores is not a symptom at all: it is a single document, and the threshold
    is right about it.
    """

    submodel: ProbabilisticModel
    threshold: float = 0.5
    rank_quantile: float = 0.75
    crowded_share: float = 0.6
    minimum_spread: float = 0.05
    name: str = field(default="adaptive")

    def fit(self, streams: Sequence[Stream]) -> None:
        if callable(getattr(self.submodel, "fit", None)):
            self.submodel.fit(streams)  # type: ignore[attr-defined]

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        return self.submodel.predict_probs(stream)

    def used_ranking(self, stream: Stream) -> bool:
        """Whether this stream falls back to the ranking, for reporting."""
        probs = self.predict_probs(stream)[1:]
        if not probs:
            return False
        return self._degenerate(probs)

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        probs = self.predict_probs(stream)
        rest = probs[1:]
        if not rest:
            return (True,)
        if not self._degenerate(rest):
            return (True, *(p > self.threshold for p in rest))
        ranks = rank_quantiles(rest)
        return (True, *(r >= self.rank_quantile for r in ranks))

    def _degenerate(self, probs: Sequence[float]) -> bool:
        share = sum(p > self.threshold for p in probs) / len(probs)
        spread = max(probs) - min(probs)
        return share > self.crowded_share or spread < self.minimum_spread
