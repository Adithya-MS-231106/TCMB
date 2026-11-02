
"""Exact total-budget matching for adaptive batch allocations."""

from typing import List


def match_total_budget(
    proposed_budgets: List[int],
    target_total: int,
    minimum: int,
    maximum: int,
) -> List[int]:
    """
    Adjust proposed batch budgets so their sum equals target_total.

    Constraints:
      - Each budget stays between minimum and maximum.
      - The original relative allocation is preserved as much
        as possible through proportional scaling and rounding.
    """

    if not proposed_budgets:
        raise ValueError("proposed_budgets cannot be empty.")

    if minimum < 0 or maximum < minimum:
        raise ValueError("Invalid budget bounds.")

    if any(
        not isinstance(value, int) or value < minimum or value > maximum
        for value in proposed_budgets
    ):
        raise ValueError("Proposed budgets must be integers within bounds.")

    number_of_batches = len(proposed_budgets)

    lowest_total = number_of_batches * minimum
    highest_total = number_of_batches * maximum

    if not lowest_total <= target_total <= highest_total:
        raise ValueError(
            f"Target total {target_total} is infeasible. "
            f"Allowed range: {lowest_total} to {highest_total}."
        )

    budgets = list(proposed_budgets)
    difference = target_total - sum(budgets)

    # Distribute the difference one operation at a time.
    # Adjust batches with the greatest remaining capacity first.
    while difference != 0:

        if difference > 0:
            candidates = [
                i for i in range(number_of_batches)
                if budgets[i] < maximum
            ]

            if not candidates:
                raise RuntimeError("Cannot increase budgets further.")

            candidates.sort(
                key=lambda i: (
                    budgets[i] - proposed_budgets[i],
                    -i,
                )
            )

            for i in candidates:
                if difference == 0:
                    break

                budgets[i] += 1
                difference -= 1

        else:
            candidates = [
                i for i in range(number_of_batches)
                if budgets[i] > minimum
            ]

            if not candidates:
                raise RuntimeError("Cannot decrease budgets further.")

            candidates.sort(
                key=lambda i: (
                    proposed_budgets[i] - budgets[i],
                    i,
                )
            )

            for i in candidates:
                if difference == 0:
                    break

                budgets[i] -= 1
                difference += 1

    assert sum(budgets) == target_total
    assert all(minimum <= value <= maximum for value in budgets)

    return budgets