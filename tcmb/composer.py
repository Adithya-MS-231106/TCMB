"""Greedy token-pair merge composer for controlled experiments.

This operates on whitespace-separated token strings. It is NOT yet a drop-in
replacement for XLM-R SentencePiece and does not update pretrained embeddings.
"""
from typing import Sequence, Iterable
from .pair_scorer import PairScorer

class GreedyMergeComposer:
    def __init__(self, merge_rules=None):
        self.merge_rules = [tuple(x) for x in (merge_rules or [])]

    def fit(self, token_sequences: Iterable[Sequence[str]], max_merges: int = 100, min_frequency: int = 2):
        scorer = PairScorer(min_frequency=min_frequency).fit(token_sequences)
        self.merge_rules = [tuple(item["pair"]) for item in scorer.propose(max_merges)]
        return self

    def compose(self, tokens: Sequence[str], budget: int = 0):
        output = list(tokens)
        applied = []
        for left, right in self.merge_rules[:max(0, budget)]:
            merged = []
            i = 0
            changed = False
            while i < len(output):
                if i + 1 < len(output) and output[i] == left and output[i + 1] == right:
                    merged.append(left + right)
                    applied.append((left, right))
                    i += 2
                    changed = True
                else:
                    merged.append(output[i])
                    i += 1
            output = merged
            if not changed:
                continue
        return output, applied

    def to_dict(self):
        return {"merge_rules": [list(pair) for pair in self.merge_rules],
                "warning": "Prototype string-level merges; not an XLM-R tokenizer retrofit."}

    @classmethod
    def from_dict(cls, data):
        return cls(data.get("merge_rules", []))
