
"""Interpretable, configurable TCMB merge-budget predictor.

Version 1 is a heuristic research baseline, not a trained model.
Typology and fragmentation weights must be calibrated experimentally.
"""

from .typological_features import sentence_complexity


class TypologyConditionedBudgetPredictor:

    def __init__(
        self,
        min_budget: int = 0,
        max_budget: int = 20,
        base_budget: int = 5,
        fragmentation_weight: float = 0.0,
        complexity_weight: float = 0.0,
    ):
        if min_budget < 0 or max_budget < min_budget:
            raise ValueError("Invalid budget range.")

        if not min_budget <= base_budget <= max_budget:
            raise ValueError("Base budget must be within budget limits.")

        self.min_budget = min_budget
        self.max_budget = max_budget
        self.base_budget = base_budget
        self.fragmentation_weight = fragmentation_weight
        self.complexity_weight = complexity_weight

    def predict(
        self,
        language: str,
        text: str,
        fragmentation_score: float = 0.0,
        pair_candidates: int = 100,
    ) -> int:

        if pair_candidates < 0:
            raise ValueError("pair_candidates cannot be negative.")

        if fragmentation_score < 0:
            raise ValueError("fragmentation_score cannot be negative.")

        complexity = sentence_complexity(text)

        # Normalize sentence length to a 0–1 range.
        length_score = min(complexity["char_count"] / 200.0, 1.0)

        # Initial experimental budget formula.
        adjustment = (
            self.fragmentation_weight * fragmentation_score
            + self.complexity_weight * length_score
        )

        budget = round(self.base_budget + adjustment)

        budget = max(self.min_budget, min(self.max_budget, budget))

        # Cannot apply more merges than available candidate pairs.
        return min(budget, pair_candidates)

    def predict_batch(
        self,
        language: str,
        texts,
        fragmentation_scores=None,
        pair_candidates: int = 100,
    ):

        if fragmentation_scores is None:
            fragmentation_scores = [0.0] * len(texts)

        if len(texts) != len(fragmentation_scores):
            raise ValueError(
                "Texts and fragmentation_scores must have equal lengths."
            )

        return [
            self.predict(
                language=language,
                text=text,
                fragmentation_score=score,
                pair_candidates=pair_candidates,
            )
            for text, score in zip(texts, fragmentation_scores)
        ]
