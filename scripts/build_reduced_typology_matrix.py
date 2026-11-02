
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "typology" / "tcmb_typology_features_v1.csv"
OUTPUT = ROOT / "data" / "typology" / "tcmb_typology_features_v2.csv"

SELECTED = [
    "S_NEGATIVE_AFFIX",
    "S_NEGATIVE_WORD",
    "S_NEGATIVE_WORD_AFTER_OBJECT",
    "S_NEGATIVE_WORD_BEFORE_OBJECT",
    "S_OBJECT_BEFORE_VERB",
    "S_POLARQ_AFFIX",
    "S_POLARQ_MARK_FINAL",
    "S_POLARQ_MARK_INITIAL",
    "S_POLARQ_WORD",
    "S_POSSESSOR_AFTER_NOUN",
]

df = pd.read_csv(INPUT)

reduced = df[["language_code"] + SELECTED].copy()

if reduced.isnull().any().any():
    raise ValueError("Missing values found in reduced matrix.")

if not reduced[SELECTED].isin([0, 1]).all().all():
    raise ValueError("Unexpected non-binary feature values.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
reduced.to_csv(OUTPUT, index=False)

print("=" * 60)
print("REDUCED TYPOLOGICAL MATRIX")
print("=" * 60)
print("Languages:", len(reduced))
print("Features:", len(SELECTED))
print("Dimensions:", reduced.shape)
print("\nFeature names:")
for feature in SELECTED:
    print("-", feature)

print("\nLanguage matrix:")
print(reduced.to_string(index=False))

print("\nSaved:", OUTPUT)
