import csv
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_FILE = Path("results/tables/paired_fragmentation_summary.csv")
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

languages = [row["language"] for row in rows]

mean_tokens = [
    float(row["mean_tokens"]) for row in rows
]

token_density = [
    float(row["mean_tokens_per_100_characters"])
    for row in rows
]

# Figure 1: Average tokens per sentence
plt.figure(figsize=(10, 6))
plt.bar(languages, mean_tokens)
plt.xlabel("Language")
plt.ylabel("Average tokens per sentence")
plt.title("Static XLM-R: Average Tokens per Sentence")
plt.xticks(rotation=35)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "static_xlmr_mean_tokens.png",
    dpi=300
)
plt.close()

# Figure 2: Tokens per 100 characters
plt.figure(figsize=(10, 6))
plt.bar(languages, token_density)
plt.xlabel("Language")
plt.ylabel("Tokens per 100 characters")
plt.title("Static XLM-R: Token Density by Language")
plt.xticks(rotation=35)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "static_xlmr_token_density.png",
    dpi=300
)
plt.close()

print("Baseline figures generated successfully.")
print("Saved:")
print(OUTPUT_DIR / "static_xlmr_mean_tokens.png")
print(OUTPUT_DIR / "static_xlmr_token_density.png")