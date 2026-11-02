
from pathlib import Path
import sys
import csv

from datasets import load_from_disk
from transformers import AutoTokenizer


# --------------------------------------------------
# Project paths and experiment settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "xlm_roberta"
FLORES_ROOT = ROOT / "data" / "flores_plus"

OUTPUT_DIR = ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

BATCH_SIZE = 32
MAX_MERGES = 1000


# --------------------------------------------------
# Load original Dynamic BPE implementation
# --------------------------------------------------

REFERENCE_ROOT = Path(
    r"C:\Users\vindh\AI-ML Projects\dynamic-tokenization-reference"
)

COMPAT_DIR = (
    REFERENCE_ROOT / "zett-source" / "zett-compat"
)

ORIGINAL_ZETT_DIR = (
    REFERENCE_ROOT / "zett-source" / "zett-python-only" / "zett"
)

sys.path.insert(0, str(REFERENCE_ROOT))
sys.path.insert(0, str(COMPAT_DIR))

import zett

zett.__path__.append(str(ORIGINAL_ZETT_DIR))

from tokenizations.dynamic_bpe import Dynamic_BPE
from zett.tokenizer_converters import convert_to_byte_level


# --------------------------------------------------
# Load FLORES+ dev sentences
# --------------------------------------------------

def load_dev_sentences(language):
    dataset_path = FLORES_ROOT / language / "dev"

    dataset = load_from_disk(str(dataset_path))

    return [str(text) for text in dataset["text"]]


# --------------------------------------------------
# Tokenize using the converted XLM-R tokenizer
# --------------------------------------------------

def tokenize_sentence(tokenizer, sentence):
    encoded = tokenizer(
        sentence,
        add_special_tokens=False
    )

    return tokenizer.convert_ids_to_tokens(
        encoded["input_ids"]
    )


# --------------------------------------------------
# Profile feasible merge operations in one batch
# --------------------------------------------------

def profile_batch(dynamic_bpe, original_batch):
    batch_tokens = [
        list(tokens) for tokens in original_batch
    ]

    initial_token_count = sum(
        len(tokens) for tokens in batch_tokens
    )

    merge_operations = 0
    stopped_no_pair = False

    for _ in range(MAX_MERGES):

        pair = dynamic_bpe.get_most_frequent_pair(
            batch_tokens
        )

        if pair is None or pair == "":
            stopped_no_pair = True
            break

        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"Unexpected pair returned: {pair!r}"
            )

        before_count = sum(
            len(tokens) for tokens in batch_tokens
        )

        result = dynamic_bpe.merge_pair(
            pair[0],
            pair[1],
            batch_tokens,
            ner=False,
            batch_word_ids=[]
        )

        batch_tokens = result[0]

        after_count = sum(
            len(tokens) for tokens in batch_tokens
        )

        if after_count >= before_count:
            raise RuntimeError(
                "Selected merge did not reduce token count."
            )

        merge_operations += 1

    final_token_count = sum(
        len(tokens) for tokens in batch_tokens
    )

    hit_cap = (
        merge_operations == MAX_MERGES
        and not stopped_no_pair
    )

    return {
        "initial_tokens": initial_token_count,
        "final_tokens": final_token_count,
        "merge_operations": merge_operations,
        "tokens_saved": initial_token_count - final_token_count,
        "hit_cap": hit_cap,
        "stopped_no_pair": stopped_no_pair,
    }


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    print("Loading XLM-R tokenizer...")

    original_tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        use_fast=True
    )

    tokenizer, added_bytes = convert_to_byte_level(
        original_tokenizer
    )

    print(
        f"Byte-level conversion added {added_bytes} bytes."
    )

    batch_rows = []
    language_rows = []

    for language in LANGUAGES:

        print(f"\nProfiling {language}...")

        sentences = load_dev_sentences(language)

        # Fresh Dynamic BPE instance for each language.
        dynamic_bpe = Dynamic_BPE(
            tokenizer,
            tokenizer_boundary="pretokens"
        )

        language_batches = []

        for batch_number, start in enumerate(
            range(0, len(sentences), BATCH_SIZE),
            start=1
        ):

            batch_sentences = sentences[
                start:start + BATCH_SIZE
            ]

            original_batch = [
                tokenize_sentence(tokenizer, sentence)
                for sentence in batch_sentences
            ]

            result = profile_batch(
                dynamic_bpe,
                original_batch
            )

            row = {
                "language": language,
                "batch_number": batch_number,
                "sentence_count": len(batch_sentences),
                **result,
            }

            batch_rows.append(row)
            language_batches.append(row)

            print(
                f"  Batch {batch_number:02d}: "
                f"merges={result['merge_operations']}, "
                f"tokens_saved={result['tokens_saved']}, "
                f"hit_cap={result['hit_cap']}"
            )

        total_initial = sum(
            row["initial_tokens"]
            for row in language_batches
        )

        total_final = sum(
            row["final_tokens"]
            for row in language_batches
        )

        total_merges = sum(
            row["merge_operations"]
            for row in language_batches
        )

        total_saved = sum(
            row["tokens_saved"]
            for row in language_batches
        )

        cap_batches = sum(
            row["hit_cap"]
            for row in language_batches
        )

        language_rows.append({
            "language": language,
            "batches": len(language_batches),
            "initial_tokens": total_initial,
            "final_tokens": total_final,
            "total_merge_operations": total_merges,
            "total_tokens_saved": total_saved,
            "mean_merges_per_batch": (
                total_merges / len(language_batches)
            ),
            "median_merges_per_batch": sorted(
                row["merge_operations"]
                for row in language_batches
            )[len(language_batches) // 2],
            "min_merges_per_batch": min(
                row["merge_operations"]
                for row in language_batches
            ),
            "max_merges_per_batch": max(
                row["merge_operations"]
                for row in language_batches
            ),
            "batches_hit_cap": cap_batches,
        })

    # Save batch-level results.
    batch_path = (
        OUTPUT_DIR / "dev_merge_budget_batches.csv"
    )

    with open(
        batch_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=batch_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(batch_rows)

    # Save language-level summary.
    summary_path = (
        OUTPUT_DIR / "dev_merge_budget_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=language_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(language_rows)

    print("\n" + "=" * 65)
    print("DEV MERGE-BUDGET PROFILE")
    print("=" * 65)

    for row in language_rows:
        print(
            f"{row['language']:10s} "
            f"mean={row['mean_merges_per_batch']:7.2f} "
            f"median={row['median_merges_per_batch']:5d} "
            f"min={row['min_merges_per_batch']:4d} "
            f"max={row['max_merges_per_batch']:4d} "
            f"cap_batches={row['batches_hit_cap']}"
        )

    print("\nSaved:")
    print(batch_path)
    print(summary_path)


if __name__ == "__main__":
    main()
