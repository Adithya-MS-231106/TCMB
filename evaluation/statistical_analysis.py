import csv
import random
import statistics
from pathlib import Path

INPUT_FILE = Path("results/tables/paired_fragmentation.csv")
OUTPUT_FILE = Path(
    "results/tables/static_xlmr_statistical_analysis.csv"
)

LANGUAGES = [
    "eng_Latn",
    "tam_Taml",
    "tel_Telu",
    "kan_Knda",
    "mal_Mlym",
    "hin_Deva",
    "cmn_Hans",
    "jpn_Jpan",
]

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Load sentence-level data
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

# Organize token counts by language and sentence ID
data = {language: {} for language in LANGUAGES}

for row in rows:
    language = row["language"]
    sentence_id = row["sentence_id"]
    token_count = int(row["token_count"])

    data[language][sentence_id] = token_count

# Verify alignment
reference_ids = set(data["eng_Latn"])

for language in LANGUAGES:
    if set(data[language]) != reference_ids:
        raise ValueError(
            f"Sentence IDs do not match for {language}"
        )

english_counts = data["eng_Latn"]
english_values = list(english_counts.values())
english_mean = statistics.mean(english_values)

results = []

for language in LANGUAGES:

    values = list(data[language].values())

    mean_tokens = statistics.mean(values)
    std_tokens = statistics.stdev(values)

    if language == "eng_Latn":
        mean_difference = 0.0
        relative_difference = 0.0
        difference_std = 0.0
        ci_lower = 0.0
        ci_upper = 0.0

    else:
        # Paired differences: same sentence IDs
        differences = [
            data[language][sid] - english_counts[sid]
            for sid in sorted(reference_ids)
        ]

        mean_difference = statistics.mean(differences)
        difference_std = statistics.stdev(differences)

        relative_difference = (
            mean_difference / english_mean
        ) * 100

        # Paired bootstrap confidence interval
        bootstrap_means = []

        for _ in range(BOOTSTRAP_ITERATIONS):
            sample = random.choices(
                differences,
                k=len(differences)
            )

            bootstrap_means.append(
                statistics.mean(sample)
            )

        bootstrap_means.sort()

        ci_lower = bootstrap_means[
            int(0.025 * BOOTSTRAP_ITERATIONS)
        ]

        ci_upper = bootstrap_means[
            int(0.975 * BOOTSTRAP_ITERATIONS)
        ]

    results.append({
        "language": language,
        "sentence_count": len(values),
        "mean_tokens": round(mean_tokens, 4),
        "standard_deviation": round(std_tokens, 4),
        "mean_difference_vs_english": round(
            mean_difference, 4
        ),
        "difference_standard_deviation": round(
            difference_std, 4
        ),
        "relative_difference_percent": round(
            relative_difference, 2
        ),
        "bootstrap_95ci_lower": round(ci_lower, 4),
        "bootstrap_95ci_upper": round(ci_upper, 4),
    })

# Save results
with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

# Display summary
print("\nSTATIC XLM-R STATISTICAL ANALYSIS")
print("-" * 75)

for result in results:
    print(
        f"{result['language']:10} "
        f"Mean={result['mean_tokens']:7.2f} "
        f"Difference={result['mean_difference_vs_english']:7.2f} "
        f"95% CI=[{result['bootstrap_95ci_lower']:.2f}, "
        f"{result['bootstrap_95ci_upper']:.2f}]"
    )

print("\nAnalysis completed.")
print(f"Results saved to: {OUTPUT_FILE}")