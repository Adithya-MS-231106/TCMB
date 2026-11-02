
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

STATIC_FILE = TABLE_DIR / "static_tokenizer_comparison.csv"
DYNAMIC_FILE = TABLE_DIR / "dynamic_bpe_summary.csv"

OUTPUT_FILE = TABLE_DIR / "master_baseline_comparison.csv"


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main():

    static_rows = load_csv(STATIC_FILE)
    dynamic_rows = load_csv(DYNAMIC_FILE)

    static_data = {
        row["language"]: row
        for row in static_rows
    }

    dynamic_data = {
        row["language_code"]: row
        for row in dynamic_rows
    }

    results = []

    for language_code, dynamic in dynamic_data.items():

        if language_code not in static_data:
            raise ValueError(
                f"Missing static baseline for {language_code}"
            )

        static = static_data[language_code]

        original_mean = float(static["original_xlmr_mean"])
        converted_mean = float(static["converted_xlmr_mean"])
        dynamic_mean = float(dynamic["mean_dynamic_tokens"])

        if int(static["sentences"]) != int(dynamic["sentence_count"]):
            raise ValueError(
                f"Sentence count mismatch for {language_code}"
            )

        # Verify Dynamic BPE starts from the converted baseline.
        dynamic_start = float(dynamic["mean_original_tokens"])

        if abs(dynamic_start - converted_mean) > 0.0001:
            raise ValueError(
                f"Starting-token mismatch for {language_code}: "
                f"Dynamic BPE={dynamic_start}, "
                f"converted baseline={converted_mean}"
            )

        # Fair reduction: Dynamic BPE compared with converted XLM-R.
        reduction_vs_converted = (
            (converted_mean - dynamic_mean)
            / converted_mean
        ) * 100

        # Diagnostic only: comparison with original XLM-R.
        reduction_vs_original = (
            (original_mean - dynamic_mean)
            / original_mean
        ) * 100

        results.append({
            "language_code": language_code,
            "language": dynamic["language"],
            "sentences": int(dynamic["sentence_count"]),
            "original_xlmr_mean": round(original_mean, 4),
            "converted_xlmr_mean": round(converted_mean, 4),
            "dynamic_bpe_mean": round(dynamic_mean, 4),
            "conversion_change_percent": round(
                float(static["conversion_token_change_percent"]), 4
            ),
            "dynamic_reduction_vs_converted_percent": round(
                reduction_vs_converted, 4
            ),
            "dynamic_reduction_vs_original_percent_diagnostic": round(
                reduction_vs_original, 4
            ),
            "total_merges": int(dynamic["total_merges"]),
            "preservation_rate": float(dynamic["preservation_rate"]),
        })

    # Save results
    fieldnames = list(results[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("=" * 75)
    print("MASTER BASELINE COMPARISON")
    print("=" * 75)

    for row in results:
        print(
            f"{row['language']:<12} "
            f"Original={row['original_xlmr_mean']:>7.2f} | "
            f"Converted={row['converted_xlmr_mean']:>7.2f} | "
            f"Dynamic={row['dynamic_bpe_mean']:>7.2f} | "
            f"Reduction={row['dynamic_reduction_vs_converted_percent']:>6.2f}%"
        )

    print("\nAll baseline consistency checks passed.")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
