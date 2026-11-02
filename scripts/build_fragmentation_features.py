
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "tables" / "dynamic_bpe_sentence_results.csv"
OUTPUT_FILE = ROOT / "results" / "tables" / "sentence_fragmentation_features.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    required = {
        "language_code",
        "sentence_id",
        "original_tokens",
        "dynamic_tokens",
        "original_text",
        "text_preserved",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["original_text"] = df["original_text"].fillna("").astype(str)

    # Character count, excluding whitespace.
    df["character_count"] = df["original_text"].apply(
        lambda text: sum(not char.isspace() for char in text)
    )

    denominator = df["character_count"].clip(lower=1)

    # Fragmentation before and after Dynamic BPE.
    df["baseline_fragmentation"] = (
        df["original_tokens"] / denominator * 100
    )

    df["dynamic_fragmentation"] = (
        df["dynamic_tokens"] / denominator * 100
    )

    df["fragmentation_change"] = (
        df["baseline_fragmentation"] - df["dynamic_fragmentation"]
    )

    # Keep the output focused.
    output_columns = [
        "language_code",
        "sentence_id",
        "character_count",
        "original_tokens",
        "dynamic_tokens",
        "baseline_fragmentation",
        "dynamic_fragmentation",
        "fragmentation_change",
        "text_preserved",
    ]

    result = df[output_columns].copy()

    if result.isna().any().any():
        raise ValueError("Missing values found in output.")

    if (result["character_count"] < 0).any():
        raise ValueError("Invalid character count.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("SENTENCE FRAGMENTATION FEATURES")
    print(f"Observations: {len(result)}")
    print(f"Languages: {result['language_code'].nunique()}")
    print(f"Text preservation: {result['text_preserved'].mean():.2%}")
    print("\nLanguage-level summary:")

    summary = result.groupby("language_code").agg(
        mean_baseline_fragmentation=("baseline_fragmentation", "mean"),
        mean_dynamic_fragmentation=("dynamic_fragmentation", "mean"),
        mean_fragmentation_change=("fragmentation_change", "mean"),
    )

    print(summary.round(3).to_string())
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
