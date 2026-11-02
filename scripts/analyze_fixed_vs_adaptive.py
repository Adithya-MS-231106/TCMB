
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, spearmanr

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "tables" / "dev_fixed_vs_adaptive_batches.csv"
OUTPUT = ROOT / "results" / "tables"

with open(INPUT, encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

languages = sorted(set(row["language"] for row in rows))

rng = np.random.default_rng(42)
BOOTSTRAP_ITERATIONS = 10000

results = []
correlation_results = []

print("PAIRED FIXED VS ADAPTIVE ANALYSIS")
print("=" * 85)

for language in languages:

    language_rows = [
        row for row in rows
        if row["language"] == language
    ]

    fixed = {
        int(row["batch_index"]): row
        for row in language_rows
        if row["policy"] == "fixed"
    }

    adaptive = {
        int(row["batch_index"]): row
        for row in language_rows
        if row["policy"] == "adaptive"
    }

    batch_ids = sorted(fixed)

    if batch_ids != sorted(adaptive):
        raise ValueError(f"Batch mismatch: {language}")

    fixed_reduction = np.array([
        float(fixed[i]["reduction_percent"])
        for i in batch_ids
    ])

    adaptive_reduction = np.array([
        float(adaptive[i]["reduction_percent"])
        for i in batch_ids
    ])

    differences = adaptive_reduction - fixed_reduction

    mean_difference = float(np.mean(differences))

    # Paired bootstrap confidence interval.
    bootstrap_means = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = rng.choice(
            differences,
            size=len(differences),
            replace=True,
        )
        bootstrap_means.append(np.mean(sample))

    ci_lower, ci_upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    # Paired Wilcoxon signed-rank test.
    if np.allclose(differences, 0):
        statistic = 0.0
        p_value = 1.0
    else:
        test = wilcoxon(
            adaptive_reduction,
            fixed_reduction,
            alternative="two-sided",
            zero_method="wilcox",
        )
        statistic = float(test.statistic)
        p_value = float(test.pvalue)

    # Check whether budget matching preserved the
    # relationship between density and final budget.
    densities = np.array([
        float(adaptive[i]["token_density"])
        for i in batch_ids
    ])

    matched_budgets = np.array([
        float(adaptive[i]["allowed_operations"])
        for i in batch_ids
    ])

    rho, rho_p = spearmanr(
        densities,
        matched_budgets,
    )

    results.append({
        "language": language,
        "batch_count": len(batch_ids),
        "mean_fixed_reduction": np.mean(fixed_reduction),
        "mean_adaptive_reduction": np.mean(adaptive_reduction),
        "mean_difference_pp": mean_difference,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "wilcoxon_statistic": statistic,
        "raw_p_value": p_value,
    })

    correlation_results.append({
        "language": language,
        "density_budget_spearman_rho": float(rho),
        "correlation_p_value": float(rho_p),
    })

# Holm correction across eight language-level tests.
ordered = sorted(
    range(len(results)),
    key=lambda i: results[i]["raw_p_value"],
)

adjusted = [0.0] * len(results)
running_max = 0.0
number_of_tests = len(results)

for rank, index in enumerate(ordered):
    adjusted_value = min(
        1.0,
        (number_of_tests - rank)
        * results[index]["raw_p_value"],
    )

    running_max = max(running_max, adjusted_value)
    adjusted[index] = running_max

for index, result in enumerate(results):
    result["holm_adjusted_p"] = adjusted[index]
    result["significant_holm_0_05"] = adjusted[index] < 0.05

print("\nLANGUAGE-LEVEL RESULTS")
print("-" * 105)

for result in results:
    print(
        f"{result['language']:10s} "
        f"Difference={result['mean_difference_pp']:+.4f} pp "
        f"CI=[{result['bootstrap_ci_lower']:+.4f}, "
        f"{result['bootstrap_ci_upper']:+.4f}] "
        f"Raw p={result['raw_p_value']:.5g} "
        f"Holm p={result['holm_adjusted_p']:.5g} "
        f"Significant={result['significant_holm_0_05']}"
    )

print("\nDENSITY VS MATCHED BUDGET")
print("-" * 85)

for result in correlation_results:
    print(
        f"{result['language']:10s} "
        f"Spearman rho={result['density_budget_spearman_rho']:+.4f} "
        f"p={result['correlation_p_value']:.5g}"
    )

# Save paired comparison results.
for filename, data in [
    ("fixed_vs_adaptive_statistics.csv", results),
    ("adaptive_budget_correlations.csv", correlation_results),
]:
    path = OUTPUT / filename

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=data[0].keys(),
        )
        writer.writeheader()
        writer.writerows(data)

    print(f"\nSaved: {path}")

print("\nStatistical analysis completed.")
