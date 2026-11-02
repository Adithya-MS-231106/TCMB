
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "typology" / "uriel_features_long.csv"
OUTPUT = ROOT / "data" / "typology" / "universal_syntax_wals.csv"

values = defaultdict(dict)

with INPUT.open("r", encoding="utf-8-sig", newline="") as file:
    for row in csv.DictReader(file):
        if row["feature_set"] != "syntax_wals":
            continue

        if row["is_missing"].strip().lower() == "true":
            continue

        values[row["feature_code"]][row["language_code"]] = (
            row["feature_value"]
        )

languages = [
    "eng", "tam", "tel", "kan",
    "mal", "hin", "cmn", "jpn"
]

universal = {
    feature: language_values
    for feature, language_values in values.items()
    if all(lang in language_values for lang in languages)
}

print("Universal syntax_wals features:", len(universal))

constant = []
variable = []

for feature, language_values in sorted(universal.items()):
    unique_values = set(language_values.values())

    if len(unique_values) == 1:
        constant.append(feature)
    else:
        variable.append(feature)

print("\nConstant features:", len(constant))
for feature in constant:
    print(" ", feature)

print("\nVariable features:", len(variable))
for feature in variable:
    print(" ", feature, "=>", language_values if False else
          {lang: universal[feature][lang] for lang in languages})

# Save a wide-format CSV for the universal features.
with OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["feature_code"] + languages)

    for feature in sorted(universal):
        writer.writerow(
            [feature] + [universal[feature][lang] for lang in languages]
        )

print("\nSaved:", OUTPUT)
