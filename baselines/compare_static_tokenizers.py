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

LANGUAGES = [
    "eng_Latn", "tam_Taml", "tel_Telu", "kan_Knda",
    "mal_Mlym", "hin_Deva", "cmn_Hans", "jpn_Jpan",
]

# Import the original reference converter
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

    # Load two independent tokenizer instances
    original_tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    tokenizer_to_convert = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    converted_tokenizer, added_bytes = convert_to_byte_level(
        tokenizer_to_convert
    )

    print(f"Added byte tokens: {added_bytes}")

    results = []

    for language in LANGUAGES:

        print(f"\nProcessing {language}...")

        dataset = load_from_disk(
            str(DATA_ROOT / language / "devtest")
        )

        original_total = 0
        converted_total = 0
        sentence_count = len(dataset)

        for text in dataset["text"]:

            original_ids = original_tokenizer(
                text,
                add_special_tokens=False
            )["input_ids"]

            converted_ids = converted_tokenizer(
                text,
                add_special_tokens=False
            )["input_ids"]

            original_total += len(original_ids)
            converted_total += len(converted_ids)

        original_mean = original_total / sentence_count
        converted_mean = converted_total / sentence_count

        conversion_change = (
            (converted_mean - original_mean)
            / original_mean
        ) * 100

        row = {
            "language": language,
            "sentences": sentence_count,
            "original_xlmr_mean": round(original_mean, 4),
            "converted_xlmr_mean": round(converted_mean, 4),
            "conversion_token_change_percent": round(
                conversion_change, 4
            ),
        }

        results.append(row)

        print(f"Original XLM-R: {original_mean:.4f}")
        print(f"Converted XLM-R: {converted_mean:.4f}")
        print(f"Conversion change: {conversion_change:.2f}%")

    output_file = (
        OUTPUT_DIR / "static_tokenizer_comparison.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nComparison completed.")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()