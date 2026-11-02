import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcmb.budget_allocator import (
    FragmentationBudgetAllocator
)

ROOT = Path(__file__).resolve().parents[1]

allocator = FragmentationBudgetAllocator(
    budget_config_path=(
        ROOT / "results" / "tables"
        / "tcmb_budget_config.json"
    ),
    feature_config_path=(
        ROOT / "results" / "tables"
        / "tcmb_batch_feature_config.json"
    ),
    budget_level="moderate",
    adaptation_strength=1.0,
)

tests = [
    ("eng_Latn", 20.0),
    ("eng_Latn", 27.628),
    ("eng_Latn", 35.0),
    ("tam_Taml", 25.0),
    ("tam_Taml", 30.159),
    ("tam_Taml", 40.0),
]

print("FRAGMENTATION-ADAPTIVE BUDGET TEST")
print("=" * 65)

for language, density in tests:

    result = allocator.predict(
        language=language,
        token_density=density
    )

    print(
        f"{language:10s} "
        f"density={density:7.3f} "
        f"z={result['density_z_score']:+6.2f} "
        f"base={result['base_budget']:4d} "
        f"predicted={result['predicted_budget']:4d}"
    )

    assert (
        result["minimum_budget"]
        <= result["predicted_budget"]
        <= result["maximum_budget"]
    )

print("\nAll allocator tests passed.")