from pathlib import Path
import csv
import statistics

from datasets import load_from_disk
from transformers import AutoTokenizer

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

print("Loading XLM-R tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

sentence_results = []
summary_results = []

reference_ids = None

for language in LANGUAGES:

    print(f"Processing {language}...")

    dataset = load_from_disk(str(DATA_PATH / language))
    split = dataset["devtest"]

    ids = split["id"]
    sentences = split["text"]

    if reference_ids is None:
        reference_ids = ids
    elif ids != reference_ids:
        raise ValueError(
            f"Sentence IDs are not aligned for {language}"
        )

    token_counts = []
    tokens_per_100_chars = []

    for sentence_id, sentence in zip(ids, sentences):

        token_ids = tokenizer(
            sentence,
            add_special_tokens=False
        )["input_ids"]

        token_count = len(token_ids)
        character_count = len(sentence)

        density = (
            token_count / character_count * 100
            if character_count > 0 else 0
        )

        token_counts.append(token_count)
        tokens_per_100_chars.append(density)

        sentence_results.append({
            "sentence_id": sentence_id,
            "language": language,
            "character_count": character_count,
            "token_count": token_count,
            "tokens_per_100_characters": round(density, 4),
        })

    summary_results.append({
        "language": language,
        "sentence_count": len(sentences),
        "mean_tokens": round(statistics.mean(token_counts), 4),
        "median_tokens": round(statistics.median(token_counts), 4),
        "mean_tokens_per_100_characters": round(
            statistics.mean(tokens_per_100_chars), 4
        ),
        "median_tokens_per_100_characters": round(
            statistics.median(tokens_per_100_chars), 4
        ),
        "minimum_tokens": min(token_counts),
        "maximum_tokens": max(token_counts),
    })

    print(
        f"  Mean tokens: {statistics.mean(token_counts):.2f}"
    )

# Save sentence-level results
sentence_file = OUTPUT_PATH / "paired_fragmentation.csv"

with open(sentence_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=sentence_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(sentence_results)

# Save language-level summary
summary_file = OUTPUT_PATH / "paired_fragmentation_summary.csv"

with open(summary_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=summary_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(summary_results)

print("\n" + "=" * 50)
print("PAIRED FRAGMENTATION ANALYSIS COMPLETED")
print(f"Sentence-level results: {sentence_file}")
print(f"Language summary: {summary_file}")
print("=" * 50)