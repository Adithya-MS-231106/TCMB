import sys
import csv
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "xlm_roberta"
DATA_ROOT = PROJECT_ROOT / "data" / "flores_plus"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = {
    "eng_Latn": "English",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "hin_Deva": "Hindi",
    "cmn_Hans": "Chinese",
    "jpn_Jpan": "Japanese",
}

REFERENCE_ROOT = Path(
    r"C:\Users\vindh\AI-ML Projects\dynamic-tokenization-reference"
)

sys.path.insert(
    0,
    str(REFERENCE_ROOT / "zett-source" / "zett-compat")
)

import zett

zett.__path__.append(
    str(
        REFERENCE_ROOT
        / "zett-source"
        / "zett-python-only"
        / "zett"
    )
)

from zett.tokenizer_converters import convert_to_byte_level


def main():

    print("Loading two independent tokenizer instances...")

    # Instance 1: original tokenizer, never passed to converter
    original_tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    # Instance 2: independent copy for byte-level conversion
    tokenizer_to_convert = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    converted_tokenizer, added_bytes = convert_to_byte_level(
        tokenizer_to_convert
    )

    print(f"Added byte tokens: {added_bytes}")
    print(
        "Independent instances:",
        original_tokenizer is not converted_tokenizer
    )

    sentence_results = []
    summaries = []

    for code, name in LANGUAGES.items():

        print(f"\nChecking {name} ({code})...")

        dataset = load_from_disk(
            str(DATA_ROOT / code / "devtest")
        )

        original_counts = []
        converted_counts = []
        mismatches = 0

        for index, text in enumerate(dataset["text"]):

            # Preserve the exact dataset text; no stripping.
            original_ids = original_tokenizer(
                text,
                add_special_tokens=False
            )["input_ids"]

            converted_ids = converted_tokenizer(
                text,
                add_special_tokens=False
            )["input_ids"]

            original_count = len(original_ids)
            converted_count = len(converted_ids)

            equal = original_count == converted_count

            if not equal:
                mismatches += 1

            original_counts.append(original_count)
            converted_counts.append(converted_count)

            sentence_results.append({
                "language_code": code,
                "language": name,
                "sentence_id": index + 1,
                "original_tokens": original_count,
                "converted_tokens": converted_count,
                "token_count_equal": equal,
                "difference": converted_count - original_count,
            })

        n = len(original_counts)

        original_mean = sum(original_counts) / n
        converted_mean = sum(converted_counts) / n

        equality = (n - mismatches) / n * 100

        summaries.append({
            "language_code": code,
            "language": name,
            "sentences": n,
            "original_mean_tokens": original_mean,
            "converted_mean_tokens": converted_mean,
            "mean_difference": converted_mean - original_mean,
            "mismatches": mismatches,
            "equality_percent": equality,
        })

        print(f"Original mean: {original_mean:.4f}")
        print(f"Converted mean: {converted_mean:.4f}")
        print(f"Different sentences: {mismatches}/{n}")
        print(f"Equality: {equality:.2f}%")

    # Save sentence-level results
    sentence_path = (
        OUTPUT_DIR / "tokenizer_equivalence_sentence_results.csv"
    )

    with open(sentence_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=sentence_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(sentence_results)

    # Save language-level results
    summary_path = (
        OUTPUT_DIR / "tokenizer_equivalence_summary.csv"
    )

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summaries[0].keys()
        )
        writer.writeheader()
        writer.writerows(summaries)

    total = sum(x["sentences"] for x in summaries)
    mismatches = sum(x["mismatches"] for x in summaries)

    print("\n" + "=" * 65)
    print("CORRECTED TOKENIZER EQUIVALENCE RESULTS")
    print("=" * 65)
    print(f"Total sentences: {total}")
    print(f"Different token counts: {mismatches}")
    print(f"Equal token counts: {total - mismatches}")
    print(f"Equality: {(total - mismatches) / total * 100:.2f}%")

    print("\nSaved:")
    print(sentence_path)
    print(summary_path)


if __name__ == "__main__":
    main()