"""Frequency-based adjacent-token pair scoring for a transparent prototype."""
from collections import Counter
from typing import Iterable, Sequence

class PairScorer:
    def __init__(self, min_frequency: int = 2):
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1.")
        self.min_frequency = min_frequency
        self.counts = Counter()

    def fit(self, token_sequences: Iterable[Sequence[str]]):
        self.counts.clear()
        for seq in token_sequences:
            self.counts.update(zip(seq, seq[1:]))
        return self

    def score(self, pair):
        return float(self.counts[tuple(pair)])

    def ranked_pairs(self):
        return [
            {"pair": list(pair), "frequency": int(freq), "score": float(freq)}
            for pair, freq in self.counts.most_common()
            if freq >= self.min_frequency
        ]

    def propose(self, max_pairs: int = 100):
        return self.ranked_pairs()[:max_pairs]
