
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcmb.budget_matching import match_total_budget


print("EXACT BUDGET MATCHING TEST")
print("=" * 65)

# Test 1: Increase total
proposed = [60, 70, 80, 90]
matched = match_total_budget(
    proposed_budgets=proposed,
    target_total=320,
    minimum=40,
    maximum=100,
)

print("Test 1:", proposed, "->", matched)
assert sum(matched) == 320

# Test 2: Decrease total
proposed = [90, 85, 80, 75]
matched = match_total_budget(
    proposed_budgets=proposed,
    target_total=280,
    minimum=40,
    maximum=100,
)

print("Test 2:", proposed, "->", matched)
assert sum(matched) == 280

# Test 3: Preserve budgets when already matched
proposed = [50, 60, 70, 80]
matched = match_total_budget(
    proposed_budgets=proposed,
    target_total=260,
    minimum=40,
    maximum=100,
)

print("Test 3:", proposed, "->", matched)
assert matched == proposed

# Test 4: Reject impossible target
try:
    match_total_budget(
        proposed_budgets=[50, 60],
        target_total=250,
        minimum=40,
        maximum=100,
    )
    raise AssertionError("Expected ValueError was not raised.")

except ValueError:
    print("Test 4: Infeasible target correctly rejected.")

# Test 5: Respect bounds
proposed = [100, 100, 100, 100]
matched = match_total_budget(
    proposed_budgets=proposed,
    target_total=350,
    minimum=50,
    maximum=100,
)

print("Test 5:", proposed, "->", matched)
assert sum(matched) == 350
assert all(50 <= value <= 100 for value in matched)

print("\nAll budget matching tests passed.")
