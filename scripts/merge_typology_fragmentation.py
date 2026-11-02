
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"

TYPOLOGY_FILE = (
    ROOT / "data" / "typology" /
    "tcmb_typology_features_v2.csv"
)

ORIGINAL_FILE = RESULTS / "paired_fragmentation_summary.csv"
MASTER_FILE = RESULTS / "master_baseline_comparison.csv"

OUTPUT = RESULTS / "typology_fragmentation_analysis.csv"

# Map language names and ISO-style codes to a common code.
LANGUAGE_MAP = {
    "english": "eng",
    "tamil": "tam",
    "telugu": "tel",
    "kannada": "kan",
    "malayalam": "mal",
    "hindi": "hin",
    "chinese": "cmn",
    "japanese": "jpn",
    "eng_latn": "eng",
    "tam_taml": "tam",
    "tel_telu": "tel",
    "kan_knda": "kan",
    "mal_mlym": "mal",
    "hin_deva": "hin",
    "cmn_hans": "cmn",
    "zho_hans": "cmn",
    "jpn_jpan": "jpn",
}

def normalize_language(value):
    key = str(value).strip().lower()
    return LANGUAGE_MAP.get(key, key)


def main():
    typology = pd.read_csv(TYPOLOGY_FILE)
    original = pd.read_csv(ORIGINAL_FILE)
    master = pd.read_csv(MASTER_FILE)

    print("Original language values:",
          original["language"].unique().tolist())

    print("Master language codes:",
          master["language_code"].unique().tolist())

    typology["language_code"] = (
        typology["language_code"].map(normalize_language)
    )

    original["language_code"] = (
        original["language"].map(normalize_language)
    )

    master["language_code"] = (
        master["language_code"].map(normalize_language)
    )

    # Keep only the useful original fragmentation columns.
    original = original[
        [
            "language_code",
            "mean_tokens",
            "median_tokens",
            "mean_tokens_per_100_characters",
            "median_tokens_per_100_characters",
        ]
    ]

    # Keep the master baseline measurements.
    master = master[
        [
            "language_code",
            "original_xlmr_mean",
            "converted_xlmr_mean",
            "dynamic_bpe_mean",
            "conversion_change_percent",
            "dynamic_reduction_vs_converted_percent",
            "total_merges",
            "preservation_rate",
        ]
    ]

    # Validate unique language keys before merging.
    for name, df in [
        ("typology", typology),
        ("original", original),
        ("master", master),
    ]:
        if df["language_code"].duplicated().any():
            raise ValueError(
                f"Duplicate language codes in {name} dataset."
            )

    merged = typology.merge(
        original,
        on="language_code",
        how="inner"
    )

    merged = merged.merge(
        master,
        on="language_code",
        how="inner"
    )

    expected = {
        "eng", "tam", "tel", "kan",
        "mal", "hin", "cmn", "jpn"
    }

    actual = set(merged["language_code"])

    if actual != expected:
        raise ValueError(
            f"Language mismatch. Missing: {expected - actual}; "
            f"Unexpected: {actual - expected}"
        )

    if merged.isnull().any().any():
        print("\nWarning: missing values detected.")
        print(merged.isnull().sum()[merged.isnull().sum() > 0])
    else:
        print("\nNo missing values detected.")

    merged = merged.sort_values("language_code")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)

    print("\n" + "=" * 65)
    print("TYPOLOGY + FRAGMENTATION DATASET")
    print("=" * 65)
    print("Languages:", len(merged))
    print("Columns:", len(merged.columns))
    print("\nPreview:")
    print(merged.to_string(index=False))
    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
