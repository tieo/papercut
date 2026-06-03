from __future__ import annotations

from dataclasses import dataclass, field

from papercut.streams.resolver import PageResolver
from papercut.streams.types import Stream


def _ngrams(text: str, n: int) -> set[str]:
    text = text.strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


@dataclass
class TextSimilarityBaseline:
    """Boundary where consecutive pages are textually dissimilar.

    Character n-gram Jaccard between page N-1 and page N. Boundary when the
    similarity is at or below `threshold`. No training, no OCR, language-
    agnostic (operates on raw characters). Stronger than trivial baselines on
    real corpora because intra-document pages share letterheads, footers,
    fonts, and section vocabulary.
    """

    resolver: PageResolver
    n: int = 4
    threshold: float = 0.15
    name: str = field(default="text_similarity")

    def predict_boundaries(self, stream: Stream) -> tuple[bool, ...]:
        probs = self.predict_probs(stream)
        return (True, *(p > 1 - self.threshold for p in probs[1:]))

    def predict_probs(self, stream: Stream) -> tuple[float, ...]:
        """Per-page boundary probability: 1 - Jaccard(prev, curr).

        High dissimilarity reads as a high boundary probability. Page 0 is
        fixed at 1.0 by convention.
        """
        page_ngrams = [_ngrams(self.resolver.text(p), self.n) for p in stream.pages]
        probs: list[float] = [1.0]
        for i in range(1, len(stream)):
            sim = _jaccard(page_ngrams[i - 1], page_ngrams[i])
            probs.append(1.0 - sim)
        return tuple(probs)
