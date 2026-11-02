
"""Batch-level merge-budget allocation for TCMB.

Version 1: fragmentation-adaptive allocation.
Typological conditioning is deliberately kept separate
for controlled ablation experiments.
"""

import json
import math
from pathlib import Path


class FragmentationBudgetAllocator:

    def __init__(
        self,
        budget_config_path,
        feature_config_path,
        budget_level="moderate",
        adaptation_strength=1.0,
    ):

        self.budget_config_path = Path(budget_config_path)
        self.feature_config_path = Path(feature_config_path)

        if budget_level not in {
            "conservative", "moderate", "aggressive"
        }:
            raise ValueError("Invalid budget level.")

        if adaptation_strength < 0:
            raise ValueError("Adaptation strength cannot be negative.")

        with open(
            self.budget_config_path,
            encoding="utf-8"
        ) as file:
            self.budget_config = json.load(file)

        with open(
            self.feature_config_path,
            encoding="utf-8"
        ) as file:
            self.feature_config = json.load(file)

        self.budget_level = budget_level
        self.adaptation_strength = adaptation_strength

    def predict(self, language, token_density):

        if language not in self.budget_config["languages"]:
            raise KeyError(f"Unknown language: {language}")

        if language not in self.feature_config["languages"]:
            raise KeyError(
                f"Missing feature calibration: {language}"
            )

        if token_density < 0:
            raise ValueError("Token density cannot be negative.")

        language_budget = (
            self.budget_config["languages"][language]
        )

        budgets = language_budget["budgets"]

        minimum = budgets["conservative"]
        base = budgets[self.budget_level]
        maximum = budgets["aggressive"]

        density_stats = (
            self.feature_config["languages"][language]
            ["metrics"]["token_density"]
        )

        mean = density_stats["mean"]
        std = density_stats["std"]

        if std == 0:
            z_score = 0.0
        else:
            z_score = (token_density - mean) / std

        # Limit extreme deviations.
        z_score = max(-2.0, min(2.0, z_score))

        # Adaptive adjustment around the selected base budget.
        half_range = (maximum - minimum) / 2.0

        adjustment = (
            self.adaptation_strength
            * z_score
            * half_range
            / 2.0
        )

        predicted_budget = base + adjustment

        predicted_budget = max(
            minimum,
            min(maximum, predicted_budget)
        )

        return {
            "language": language,
            "token_density": token_density,
            "density_z_score": z_score,
            "base_budget": base,
            "predicted_budget": math.floor(
                predicted_budget + 0.5
            ),
            "minimum_budget": minimum,
            "maximum_budget": maximum,
        }