
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

with open(
    TABLES / "dev_adaptive_budget_batches.csv",
    encoding="utf-8"
) as file:
    rows = list(csv.DictReader(file))

with open(
    TABLES / "tcmb_batch_feature_config.json",
    encoding="utf-8"
) as file:
    config = json.load(file)

print("ADAPTIVE BUDGET FEATURE AUDIT")
print("=" * 90)

for language, stats in config["languages"].items():

    language_rows = [
        row for row in rows
        if row["language"] == language
    ]

    densities = [
        float(row["token_density"])
        for row in language_rows
    ]

    z_scores = [
        float(row["density_z_score"])
        for row in language_rows
    ]

    calibrated = stats["metrics"]["token_density"]

    print(f"\n{language}")
    print(f"  Calibration mean: {calibrated['mean']:.4f}")
    print(f"  Calibration std:  {calibrated['std']:.4f}")
    print(f"  Profile mean:     {mean(densities):.4f}")
    print(f"  Profile std:      {pstdev(densities):.4f}")
    print(f"  Mean z-score:     {mean(z_scores):+.4f}")
    print(
        f"  Z-score range:    "
        f"{min(z_scores):+.4f} to {max(z_scores):+.4f}"
    )