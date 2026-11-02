
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "typology" / "tcmb_typology_features_v1.csv"
OUTPUT = ROOT / "results" / "tables" / "typology_feature_correlations.csv"

df = pd.read_csv(INPUT)

features = df.drop(columns=["language_code"])

# Pearson correlation is equivalent to the phi coefficient
# for pairs of binary variables.
correlations = features.corr(method="pearson")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
correlations.to_csv(OUTPUT)

print("=" * 65)
print("TYPOLOGICAL FEATURE REDUNDANCY ANALYSIS")
print("=" * 65)

print("\nHighly correlated feature pairs (|r| >= 0.80):")

found = False

for i in range(len(features.columns)):
    for j in range(i + 1, len(features.columns)):
        feature_a = features.columns[i]
        feature_b = features.columns[j]
        value = correlations.loc[feature_a, feature_b]

        if abs(value) >= 0.80:
            print(
                f"{feature_a} <-> {feature_b}: {value:.3f}"
            )
            found = True

if not found:
    print("No pairs crossed the 0.80 threshold.")

print("\nCorrelation matrix:")
print(correlations.round(2).to_string())

print("\nSaved:", OUTPUT)
print("\nIMPORTANT: These correlations are exploratory.")
print("Only eight languages are represented.")

