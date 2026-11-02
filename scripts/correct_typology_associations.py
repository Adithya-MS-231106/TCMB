
import pandas as pd
from pathlib import Path
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results" / "tables" /
    "typology_association_results.csv"
)

OUTPUT = (
    ROOT / "results" / "tables" /
    "typology_association_corrected.csv"
)

df = pd.read_csv(INPUT)

valid = df["exploratory_p_value"].notna()

df["holm_adjusted_p_value"] = float("nan")
df["significant_after_holm"] = False

adjusted = multipletests(
    df.loc[valid, "exploratory_p_value"],
    method="holm"
)

df.loc[valid, "holm_adjusted_p_value"] = adjusted[1]
df.loc[valid, "significant_after_holm"] = adjusted[0]

df = df.sort_values(
    "holm_adjusted_p_value",
    na_position="last"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("=" * 70)
print("HOLM-CORRECTED TYPOLOGY ASSOCIATIONS")
print("=" * 70)

print(
    df[
        [
            "feature",
            "spearman_rho",
            "exploratory_p_value",
            "holm_adjusted_p_value",
            "significant_after_holm",
        ]
    ].round(4).to_string(index=False)
)

print("\nSaved:", OUTPUT)
print(
    "\nInterpretation: these are exploratory associations "
    "based on eight languages."
)
