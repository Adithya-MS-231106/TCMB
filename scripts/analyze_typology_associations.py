
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results" / "tables" /
    "typology_fragmentation_analysis.csv"
)

OUTPUT = (
    ROOT / "results" / "tables" /
    "typology_association_results.csv"
)

df = pd.read_csv(INPUT)

target = "dynamic_reduction_vs_converted_percent"

features = [
    column for column in df.columns
    if column.startswith("S_")
]

results = []

for feature in features:
    x = df[feature]
    y = df[target]

    distinct_values = x.nunique()

    if distinct_values < 2:
        rho = float("nan")
        p_value = float("nan")
    else:
        rho, p_value = spearmanr(x, y)

    results.append({
        "feature": feature,
        "distinct_values": distinct_values,
        "languages_with_feature": int(x.sum()),
        "spearman_rho": rho,
        "exploratory_p_value": p_value,
    })

result_df = pd.DataFrame(results)

result_df["absolute_rho"] = (
    result_df["spearman_rho"].abs()
)

result_df = result_df.sort_values(
    "absolute_rho",
    ascending=False,
    na_position="last"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT, index=False)

print("=" * 70)
print("EXPLORATORY TYPOLOGY-FRAGMENTATION ASSOCIATIONS")
print("=" * 70)

print("Independent language observations:", len(df))
print("Typological features:", len(features))
print("Outcome:", target)

print("\nResults:")
print(result_df.round(4).to_string(index=False))

print("\nIMPORTANT:")
print("- Only eight independent languages are available.")
print("- Correlations are exploratory, not causal.")
print("- P-values are not confirmatory evidence.")
print("- Binary features with very few positive languages")
print("  have particularly limited interpretability.")

print("\nSaved:", OUTPUT)
