
import csv
from pathlib import Path

import lang2vec.lang2vec as l2v

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "typology" / "uriel_features_long.csv"

LANGUAGES = {
    "eng": "English",
    "tam": "Tamil",
    "tel": "Telugu",
    "kan": "Kannada",
    "mal": "Malayalam",
    "hin": "Hindi",
    "cmn": "Chinese",
    "jpn": "Japanese",
}

# These are available URIEL feature sets.
# They cover syntax, phonology and phonological inventories.
FEATURE_SETS = [
    "syntax_wals",
    "syntax_sswl",
    "syntax_ethnologue",
    "phonology_wals",
    "phonology_ethnologue",
    "inventory_ethnologue",
]


def main():
    rows = []

    for feature_set in FEATURE_SETS:
        print(f"\nExtracting: {feature_set}")

        result = l2v.get_features(
            list(LANGUAGES.keys()),
            feature_set,
            header=True,
        )

        feature_names = result["CODE"]

        for language_code, language_name in LANGUAGES.items():
            values = result[language_code]

            if len(values) != len(feature_names):
                raise ValueError(
                    f"Feature count mismatch for {language_code}, "
                    f"{feature_set}"
                )

            for feature_name, value in zip(feature_names, values):
                missing = value == "--"

                rows.append({
                    "language_code": language_code,
                    "language": language_name,
                    "feature_set": feature_set,
                    "feature_code": feature_name,
                    "feature_value": "" if missing else value,
                    "is_missing": missing,
                    "source": "URIEL/lang2vec",
                })

        print(f"  Features collected: {len(feature_names)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "language_code",
                "language",
                "feature_set",
                "feature_code",
                "feature_value",
                "is_missing",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nExtraction completed.")
    print(f"Total feature records: {len(rows)}")
    print(f"Saved to: {OUTPUT}")

    missing_count = sum(row["is_missing"] for row in rows)
    print(f"Missing feature values: {missing_count}")


if __name__ == "__main__":
    main()
