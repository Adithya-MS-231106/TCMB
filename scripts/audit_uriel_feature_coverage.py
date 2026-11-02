
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "typology" / "uriel_features_long.csv"

languages = {
    "eng", "tam", "tel", "kan",
    "mal", "hin", "cmn", "jpn"
}

coverage = defaultdict(set)
totals = defaultdict(set)

with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        key = (row["feature_set"], row["feature_code"])
        lang = row["language_code"]

        totals[key].add(lang)

        if row["is_missing"].strip().lower() == "false":
            coverage[key].add(lang)

print("\n" + "=" * 85)
print("FEATURE COVERAGE BY FEATURE SET")
print("=" * 85)

for feature_set in sorted({key[0] for key in totals}):
    features = [
        key for key in totals
        if key[0] == feature_set
    ]

    counts = [len(coverage[key]) for key in features]

    full = sum(count == 8 for count in counts)
    at_least_six = sum(count >= 6 for count in counts)
    at_least_four = sum(count >= 4 for count in counts)

    print(
        f"{feature_set:25s} "
        f"Total={len(features):3d} "
        f"All8={full:3d} "
        f"AtLeast6={at_least_six:3d} "
        f"AtLeast4={at_least_four:3d}"
    )

print("\n" + "=" * 85)
print("FEATURES AVAILABLE FOR AT LEAST SIX LANGUAGES")
print("=" * 85)

for key in sorted(totals):
    count = len(coverage[key])

    if count >= 6:
        print(
            f"{key[0]:25s} "
            f"{key[1]:30s} "
            f"{count}/8 languages"
        )

print("\n" + "=" * 85)
print("FEATURES AVAILABLE FOR ALL EIGHT LANGUAGES")
print("=" * 85)

full_features = [
    key for key in sorted(totals)
    if len(coverage[key]) == 8
]

print(f"Total: {len(full_features)}")
for feature_set, feature_code in full_features:
    print(feature_set, feature_code)

print("\nCoverage audit completed.")
