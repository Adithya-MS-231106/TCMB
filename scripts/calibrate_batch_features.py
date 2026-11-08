
from pathlib import Path
import sys
import csv
import json
import statistics
import unicodedata

from datasets import load_from_disk
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "xlm_roberta"
FLORES_ROOT = ROOT / "data" / "flores_plus"
OUTPUT_DIR = ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = [
    "eng_Latn", "tam_Taml", "tel_Telu", "kan_Knda",
    "mal_Mlym", "hin_Deva", "cmn_Hans", "jpn_Jpan"
]

BATCH_SIZE = 32

REFERENCE_ROOT = Path(
    r"C:\Users\vindh\AI-ML Projects\dynamic-tokenization-reference"
)

COMPAT_DIR = REFERENCE_ROOT / "zett-source" / "zett-compat"
ORIGINAL_ZETT_DIR = (
    REFERENCE_ROOT / "zett-source" / "zett-python-only" / "zett"
)

sys.path.insert(0, str(REFERENCE_ROOT))
sys.path.insert(0, str(COMPAT_DIR))

import zett
zett.__path__.append(str(ORIGINAL_ZETT_DIR))

from zett.tokenizer_converters import convert_to_byte_level


def tokenize_sentence(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=False
    )
    return tokenizer.convert_ids_to_tokens(
        encoded["input_ids"]
    )


def calculate_batch_features(tokenizer, sentences):

    token_count = 0
    non_whitespace_chars = 0
    total_chars = 0
    punctuation_count = 0

    for sentence in sentences:

        tokens = tokenize_sentence(tokenizer, sentence)
        token_count += len(tokens)

        total_chars += len(sentence)

        for char in sentence:

            if not char.isspace():
                non_whitespace_chars += 1

            if unicodedata.category(char).startswith("P"):
                punctuation_count += 1

    token_density = (
        100.0 * token_count / max(non_whitespace_chars, 1)
    )

    punctuation_ratio = (
        punctuation_count / max(total_chars, 1)
    )

    mean_sentence_length = (
        total_chars / max(len(sentences), 1)
    )

    return {
        "sentence_count": len(sentences),
        "initial_tokens": token_count,
        "non_whitespace_chars": non_whitespace_chars,
        "token_density": token_density,
        "mean_sentence_length": mean_sentence_length,
        "punctuation_ratio": punctuation_ratio,
    }


def z_score(value, mean, std):
    if std == 0:
        return 0.0
    return (value - mean) / std


def main():

    print("Loading XLM-R tokenizer...")

    original_tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        use_fast=True
    )

    tokenizer, added_bytes = convert_to_byte_level(
        original_tokenizer
    )

    print(f"Byte-level conversion added {added_bytes} bytes.")

    all_rows = []
    calibration = {}

    for language in LANGUAGES:

        print(f"\nProcessing {language}...")

        dataset = load_from_disk(
            str(FLORES_ROOT / language / "dev")
        )

        sentences = [str(x) for x in dataset["text"]]

        language_rows = []

        for batch_number, start in enumerate(
            range(0, len(sentences), BATCH_SIZE),
            start=1
        ):

            batch = sentences[start:start + BATCH_SIZE]

            features = calculate_batch_features(
                tokenizer,
                batch
            )

            row = {
                "language": language,
                "batch_number": batch_number,
                **features,
            }

            language_rows.append(row)
            all_rows.append(row)

        metric_names = [
            "token_density",
            "mean_sentence_length",
            "punctuation_ratio",
        ]

        calibration[language] = {
            "batch_count": len(language_rows),
            "metrics": {},
        }

        for metric in metric_names:

            values = [
                row[metric] for row in language_rows
            ]

            metric_mean = statistics.mean(values)
            metric_std = statistics.pstdev(values)

            calibration[language]["metrics"][metric] = {
                "mean": metric_mean,
                "std": metric_std,
                "min": min(values),
                "max": max(values),
            }

        print(
            f"  Batches: {len(language_rows)} | "
            f"Mean token density: "
            f"{calibration[language]['metrics']['token_density']['mean']:.3f}"
        )

    csv_path = OUTPUT_DIR / "dev_batch_features.csv"

    with open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = (
        OUTPUT_DIR / "tcmb_batch_feature_config.json"
    )

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "calibration_split": "FLORES+ dev",
                "batch_size": BATCH_SIZE,
                "tokenizer": "converted XLM-R byte-level",
                "normalization": "per-language z-score",
                "languages": calibration,
            },
            file,
            indent=4
        )

    print("\nBATCH FEATURE CALIBRATION COMPLETE")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
