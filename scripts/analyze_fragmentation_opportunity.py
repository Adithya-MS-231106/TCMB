
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results/tables/sentence_fragmentation_features.csv"
OUTPUT = ROOT / "results/tables/fragmentation_opportunity_analysis.csv"


def main():
    df = pd.read_csv(INPUT)

    # Reconstruct absolute token savings.
    df["tokens_saved"] = (
        df["original_tokens"] - df["dynamic_tokens"]
    )

    print("FRAGMENTATION AND MERGE OPPORTUNITY")
    print(f"Observations: {len(df)}")

    rho, p_value = spearmanr(
        df["baseline_fragmentation"],
        df["tokens_saved"],
    )

    print("\nAll-language sentence-level association:")
    print(f"Spearman rho: {rho:.4f}")
    print(f"p-value: {p_value:.6g}")

    print("\nLanguage-level associations:")

    results = []

    for language, group in df.groupby("language_code"):
        if group["baseline_fragmentation"].nunique() < 2:
            continue

        r, p = spearmanr(
            group["baseline_fragmentation"],
            group["tokens_saved"],
        )

        results.append({
            "language_code": language,
            "observations": len(group),
            "spearman_rho": r,
            "p_value": p,
        })

        print(
            f"{language:10s} "
            f"rho={r:+.4f} "
            f"p={p:.6g}"
        )

    pd.DataFrame(results).to_csv(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()