
import sys
import csv
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcmb.budget_matching import match_total_budget

TABLES = ROOT / "results" / "tables"

input_path = TABLES / "dev_adaptive_budget_batches.csv"
config_path = TABLES / "tcmb_budget_config.json"

with open(input_path, encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

with open(config_path, encoding="utf-8") as file:
    config = json.load(file)

grouped = defaultdict(list)

for row in rows:
    grouped[row["language"]].append(row)

output_rows = []
summary_rows = []

print("MATCHED DEVELOPMENT BUDGETS")
print("=" * 85)

for language, language_rows in grouped.items():

    language_rows.sort(
        key=lambda row: int(row["batch_index"])
    )

    budget_info = config["languages"][language]["budgets"]

    minimum = int(budget_info["conservative"])
    moderate = int(budget_info["moderate"])
    maximum = int(budget_info["aggressive"])

    batch_count = len(language_rows)
    target_total = batch_count * moderate

    proposed = [
        int(row["adaptive_budget"])
        for row in language_rows
    ]

    matched = match_total_budget(
        proposed_budgets=proposed,
        target_total=target_total,
        minimum=minimum,
        maximum=maximum,
    )

    for row, matched_budget in zip(language_rows, matched):
        new_row = dict(row)
        new_row["matched_adaptive_budget"] = matched_budget
        output_rows.append(new_row)

    assert sum(matched) == target_total

    summary_rows.append({
        "language": language,
        "batch_count": batch_count,
        "fixed_budget_per_batch": moderate,
        "fixed_total_budget": target_total,
        "original_adaptive_total": sum(proposed),
        "matched_adaptive_total": sum(matched),
        "minimum_matched_budget": min(matched),
        "maximum_matched_budget": max(matched),
    })

    print(
        f"{language:10s} "
        f"Fixed={target_total:6d} "
        f"Original adaptive={sum(proposed):6d} "
        f"Matched={sum(matched):6d} "
        f"Range={min(matched)}-{max(matched)}"
    )

output_path = TABLES / "dev_matched_budget_batches.csv"
summary_path = TABLES / "dev_matched_budget_summary.csv"

with open(output_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=output_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(output_rows)

with open(summary_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=summary_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("\nAll languages matched successfully.")
print(f"Saved: {output_path}")
print(f"Saved: {summary_path}")