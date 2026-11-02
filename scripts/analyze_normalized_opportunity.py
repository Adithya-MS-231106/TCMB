
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results/tables/dynamic_bpe_sentence_results.csv"
FEATURES = ROOT / "results/tables/sentence_fragmentation_features.csv"
OUTPUT = ROOT / "results/tables/normalized_opportunity_analysis.csv"


def main():
    df = pd.read_csv(INPUT)
    features = pd.read_csv(FEATURES)

    # Join using language and sentence identifiers.
    data = df.merge(
        features[
            ["language_code", "sentence_id", "baseline_fragmentation",
             "character_count"]
        ],
        on=["language_code", "sentence_id"],
        validate="one_to_one",
    )

    print("NORMALIZED FRAGMENTATION OPPORTUNITY")
    print(f"Observations: {len(data)}")

    results = []

    for language, group in data.groupby("language_code"):
        rho, p = spearmanr(
            group["baseline_fragmentation"],
            group["reduction_percent"],
        )

        results.append({
            "language_code": language,
            "observations": len(group),
            "rho_fragmentation_vs_reduction_percent": rho,
            "p_value": p,
        })

        print(
            f"{language:10s} "
            f"rho={rho:+.4f} "
            f"p={p:.6g}"
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT, index=False)

    print("\nMean within-language correlation:")
    print(
        result_df["rho_fragmentation_vs_reduction_percent"].mean().round(4)
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
