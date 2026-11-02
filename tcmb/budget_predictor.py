"""Interpretable heuristic merge-budget predictor (not a trained neural model)."""
from .typological_features import get_profile, sentence_complexity

class TypologyConditionedBudgetPredictor:
    def __init__(self, min_budget: int = 0, max_budget: int = 20, base_budget: int = 5):
        if min_budget < 0 or max_budget < min_budget:
            raise ValueError("Invalid budget range.")
        self.min_budget, self.max_budget, self.base_budget = min_budget, max_budget, base_budget

    def predict(self, language: str, text: str, pair_candidates: int = 100):
        profile = get_profile(language)
        complexity = sentence_complexity(text)
        budget = self.base_budget
        # Conservative heuristics; these weights are experimental knobs, not learned values.
        if profile.morphology == "agglutinative":
            budget += 2
        if not profile.whitespace_word_boundaries:
            budget -= 1
        if complexity["char_count"] > 120:
            budget += 2
        if complexity["digit_ratio"] > 0.05 or complexity["punctuation_ratio"] > 0.15:
            budget -= 1
        budget = max(self.min_budget, min(self.max_budget, budget))
        return min(budget, max(0, pair_candidates))

    def predict_batch(self, language: str, texts, pair_candidates: int = 100):
        return [self.predict(language, text, pair_candidates) for text in texts]
