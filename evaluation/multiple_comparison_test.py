import csv
from pathlib import Path
from scipy.stats import ttest_1samp

INPUT_FILE = Path("results/tables/paired_fragmentation.csv")
OUTPUT_FILE = Path(
    "results/tables/holm_corrected_tests.csv"
)

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

data = {}

for row in rows:
    language = row["language"]
    sentence_id = row["sentence_id"]
    count = int(row["token_count"])

    data.setdefault(language, {})[sentence_id] = count

english = data["eng_Latn"]
languages = [lang for lang in data if lang != "eng_Latn"]

tests = []

for language in languages:

    differences = [
        data[language][sid] - english[sid]
        for sid in sorted(english)
    ]

    test = ttest_1samp(differences, popmean=0)

    tests.append({
        "language": language,
        "mean_difference": sum(differences) / len(differences),
        "t_statistic": test.statistic,
        "raw_p_value": test.pvalue,
    })

# Holm-Bonferroni correction
tests.sort(key=lambda item: item["raw_p_value"])

number_of_tests = len(tests)
previous_adjusted = 0.0

for rank, result in enumerate(tests):

    adjusted = min(
        1.0,
        (number_of_tests - rank) * result["raw_p_value"]
    )

    adjusted = max(previous_adjusted, adjusted)
    previous_adjusted = adjusted

    result["holm_adjusted_p_value"] = adjusted
    result["significant_at_0.05"] = adjusted < 0.05

# Restore language order
tests.sort(key=lambda item: item["language"])

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=tests[0].keys()
    )

    writer.writeheader()
    writer.writerows(tests)

print("\nHOLM-CORRECTED PAIRED TESTS")
print("-" * 65)

for result in tests:
    print(
        f"{result['language']:10} "
        f"Raw p={result['raw_p_value']:.6g} "
        f"Adjusted p={result['holm_adjusted_p_value']:.6g} "
        f"Significant={result['significant_at_0.05']}"
    )

print(f"\nSaved results to: {OUTPUT_FILE}")