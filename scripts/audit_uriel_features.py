
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "typology" / "uriel_features_long.csv"

by_set = defaultdict(lambda: {"total": 0, "available": 0})
by_language = defaultdict(lambda: {"total": 0, "available": 0})
by_feature = defaultdict(lambda: {"total": 0, "available": 0})

with INPUT.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        feature_set = row["feature_set"]
        language = row["language"]
        feature = row["feature_code"]

        available = row["is_missing"].strip().lower() == "false"

        for group, key in [
            (by_set, feature_set),
            (by_language, language),
            (by_feature, feature),
        ]:
            group[key]["total"] += 1
            group[key]["available"] += int(available)


def print_report(title, data):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    for name, counts in sorted(data.items()):
        total = counts["total"]
        available = counts["available"]
        missing = total - available
        coverage = 100 * available / total if total else 0

        print(
            f"{name:28s} "
            f"Available={available:4d} "
            f"Missing={missing:4d} "
            f"Coverage={coverage:6.2f}%"
        )


print_report("COVERAGE BY FEATURE SET", by_set)
print_report("COVERAGE BY LANGUAGE", by_language)

fully_covered = [
    name for name, counts in by_feature.items()
    if counts["available"] == counts["total"]
]

print("\n" + "=" * 75)
print("FEATURES AVAILABLE FOR ALL EIGHT LANGUAGES")
print("=" * 75)
print(f"Total: {len(fully_covered)}")
for feature in sorted(fully_covered):
    print(feature)

print("\nAudit completed.")
