
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "typology" / "universal_syntax_wals.csv"
OUTPUT = ROOT / "data" / "typology" / "tcmb_typology_features_v1.csv"

LANGUAGES = [
    "eng", "tam", "tel", "kan",
    "mal", "hin", "cmn", "jpn"
]

# Features identified as variable in the previous audit.
SELECTED_FEATURES = [
    "S_NEGATIVE_AFFIX",
    "S_NEGATIVE_SUFFIX",
    "S_NEGATIVE_WORD",
    "S_NEGATIVE_WORD_ADJACENT_BEFORE_VERB",
    "S_NEGATIVE_WORD_AFTER_OBJECT",
    "S_NEGATIVE_WORD_AFTER_SUBJECT",
    "S_NEGATIVE_WORD_BEFORE_OBJECT",
    "S_NEGATIVE_WORD_BEFORE_VERB",
    "S_OBJECT_AFTER_VERB",
    "S_OBJECT_BEFORE_VERB",
    "S_POLARQ_AFFIX",
    "S_POLARQ_MARK_FINAL",
    "S_POLARQ_MARK_INITIAL",
    "S_POLARQ_WORD",
    "S_POSSESSOR_AFTER_NOUN",
    "S_SOV",
    "S_SVO",
]


def main():
    feature_values = {}

    with INPUT.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            feature = row["feature_code"]

            if feature in SELECTED_FEATURES:
                feature_values[feature] = {
                    lang: float(row[lang])
                    for lang in LANGUAGES
                }

    missing = set(SELECTED_FEATURES) - set(feature_values)

    if missing:
        raise ValueError(f"Missing selected features: {missing}")

    # Verify that all selected features are binary.
    for feature, values in feature_values.items():
        for lang, value in values.items():
            if value not in (0.0, 1.0):
                raise ValueError(
                    f"Unexpected value: {feature}, {lang}, {value}"
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["language_code"] + SELECTED_FEATURES
        )

        writer.writeheader()

        for lang in LANGUAGES:
            row = {"language_code": lang}

            for feature in SELECTED_FEATURES:
                row[feature] = feature_values[feature][lang]

            writer.writerow(row)

    print("=" * 65)
    print("TCMB TYPOLOGICAL FEATURE MATRIX")
    print("=" * 65)
    print("Languages:", len(LANGUAGES))
    print("Selected features:", len(SELECTED_FEATURES))
    print("Matrix dimensions:", len(LANGUAGES), "x",
          len(SELECTED_FEATURES))
    print("Missing values: 0")
    print("All values verified as binary.")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
