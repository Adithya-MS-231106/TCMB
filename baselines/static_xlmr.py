from pathlib import Path
import csv
import re

from datasets import load_from_disk
from transformers import AutoTokenizer

# Project paths
MODEL_PATH = "models/xlm_roberta"
DATA_PATH = Path("data/flores_plus")
OUTPUT_PATH = Path("results/tables")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

LANGUAGES = [
    "eng_Latn",
    "tam_Taml",
    "tel_Telu",
    "kan_Knda",
    "mal_Mlym",
    "hin_Deva",
    "cmn_Hans",
    "jpn_Jpan",
]

# Load original XLM-R tokenizer
print("Loading XLM-R tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Tokenizer loaded successfully.\n")

results = []

for language in LANGUAGES:

    print(f"Processing {language}...")

    dataset = load_from_disk(
        str(DATA_PATH / language)
    )

    sentences = dataset["devtest"]["text"]

    total_tokens = 0
    total_words = 0
    total_characters = 0
    token_counts = []

    for sentence in sentences:

        encoded = tokenizer(
            sentence,
            add_special_tokens=False
        )

        tokens = encoded["input_ids"]
        token_count = len(tokens)

        # Basic whitespace-based word count
        words = sentence.split()

        total_tokens += token_count
        total_words += len(words)
        total_characters += len(sentence)

        token_counts.append(token_count)

    sentence_count = len(sentences)

    average_tokens = total_tokens / sentence_count

    fertility = (
        total_tokens / total_words
        if total_words > 0 else 0
    )

    characters_per_token = (
        total_characters / total_tokens
        if total_tokens > 0 else 0
    )

    maximum_tokens = max(token_counts)

    row = {
        "language": language,
        "sentences": sentence_count,
        "total_tokens": total_tokens,
        "average_tokens_per_sentence": round(average_tokens, 4),
        "total_words": total_words,
        "fertility_tokens_per_word": round(fertility, 4),
        "characters_per_token": round(characters_per_token, 4),
        "maximum_sequence_length": maximum_tokens,
    }

    results.append(row)

    print(f"  Sentences: {sentence_count}")
    print(f"  Average tokens: {average_tokens:.2f}")
    print(f"  Fertility: {fertility:.4f}")
    print(f"  Characters/token: {characters_per_token:.4f}")
    print(f"  Maximum length: {maximum_tokens}\n")


# Save results
output_file = OUTPUT_PATH / "static_xlmr_baseline.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

print("=" * 50)
print("STATIC XLM-R BASELINE COMPLETED")
print(f"Results saved to: {output_file}")
print("=" * 50)