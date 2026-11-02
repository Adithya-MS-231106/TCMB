import inspect
import sys
import csv
from pathlib import Path
from collections import defaultdict

from transformers import AutoTokenizer

# ---------------------------------------------------------
# PROJECT PATHS AND EXPERIMENT SETTINGS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "xlm_roberta"

FLORES_ROOT = PROJECT_ROOT / "data" / "flores_plus"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32

MAX_MERGES = 1000

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


# ---------------------------------------------------------
# LOAD THE ORIGINAL DYNAMIC TOKENIZATION IMPLEMENTATION
# ---------------------------------------------------------

REFERENCE_ROOT = Path(
    r"C:\Users\vindh\AI-ML Projects\dynamic-tokenization-reference"
)

COMPAT_DIR = (
    REFERENCE_ROOT
    / "zett-source"
    / "zett-compat"
)

ORIGINAL_ZETT_DIR = (
    REFERENCE_ROOT
    / "zett-source"
    / "zett-python-only"
    / "zett"
)

sys.path.insert(0, str(REFERENCE_ROOT))
sys.path.insert(0, str(COMPAT_DIR))

import zett

zett.__path__.append(str(ORIGINAL_ZETT_DIR))

from tokenizations.dynamic_bpe import Dynamic_BPE
from zett.tokenizer_converters import convert_to_byte_level


# ---------------------------------------------------------
# LOAD FLORES+ SENTENCES
# ---------------------------------------------------------


def load_flores_sentences(language_code):
    """
    Load FLORES+ devtest sentences from a Hugging Face
    Arrow dataset.

    Supports the existing dataset structure:
    data/flores_plus/<language>/devtest/
    """

    from datasets import load_from_disk

    dataset_path = (
        FLORES_ROOT
        / language_code
        / "devtest"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"FLORES+ dataset directory not found: {dataset_path}"
        )

    dataset = load_from_disk(str(dataset_path))

    # Inspect available columns and select the text column.
    print("Dataset columns:", dataset.column_names)

    possible_text_columns = [
        "sentence",
        "text",
        "translation",
    ]

    text_column = next(
        (
            column
            for column in possible_text_columns
            if column in dataset.column_names
        ),
        None
    )

    if text_column is None:
        raise ValueError(
            f"No recognized text column found. "
            f"Available columns: {dataset.column_names}"
        )

    sentences = [
        str(row[text_column]).strip()
        for row in dataset
        if row[text_column] is not None
        and str(row[text_column]).strip()
    ]

    print(
        f"Loaded {len(sentences)} sentences "
        f"for {language_code}"
    )

    return sentences


# ---------------------------------------------------------
# TOKENIZE USING THE SAME ENCODING PATH AS STATIC XLM-R
# ---------------------------------------------------------

def tokenize_sentence(tokenizer, text):
    """
    Use the tokenizer's normal encoding API with
    add_special_tokens=False.

    Convert the resulting IDs into token strings because
    Dynamic_BPE operates on token strings.

    This keeps the initial token counts comparable to
    the static XLM-R baseline.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=False
    )

    token_ids = encoded["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return tokens


# ---------------------------------------------------------
# APPLY DYNAMIC BPE MERGES TO ONE BATCH
# ---------------------------------------------------------

def count_merges(dynamic_bpe, batch_tokens, merge_budget):
    """
    Apply Dynamic BPE merges to a batch using the
    original implementation's merge_pair argument order.
    """

    merged_tokens = [
        list(tokens)
        for tokens in batch_tokens
    ]

    merges_applied = 0

    for _ in range(merge_budget):

        pair = dynamic_bpe.get_most_frequent_pair(
            merged_tokens
        )

        # The original implementation may return an
        # empty string when no valid pair remains.
        if pair is None or pair == "":
            break

        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"Unexpected pair returned by Dynamic_BPE: {pair!r}"
            )

        a, b = pair

        result = dynamic_bpe.merge_pair(
            a,
            b,
            merged_tokens,
            ner=False,
            batch_word_ids=[]
        )

        if result is not None:
            if isinstance(result, tuple):
                merged_tokens = result[0]
            else:
                merged_tokens = result

        merges_applied += 1

    return merged_tokens, merges_applied

# ---------------------------------------------------------
# DECODE AND CHECK TEXT PRESERVATION
# ---------------------------------------------------------

def decode_and_check(tokenizer, original_text, merged_tokens):
    """
    Decode the merged token sequence and compare it
    against the original sentence.

    The byte-level converter may introduce one leading
    space during decoding. Remove exactly one leading
    space only when it is present.
    """

    decoded_text = tokenizer.convert_tokens_to_string(
        merged_tokens
    )

    if decoded_text.startswith(" "):
        decoded_text = decoded_text[1:]

    preserved = decoded_text == original_text

    return decoded_text, preserved


# ---------------------------------------------------------
# PROCESS ONE LANGUAGE
# ---------------------------------------------------------

def process_language(
    language_code,
    language_name,
    tokenizer,
    dynamic_bpe
):
    print("\n" + "=" * 65)
    print(f"Processing: {language_name} ({language_code})")
    print("=" * 65)

    sentences = load_flores_sentences(language_code)

    print(f"Loaded sentences: {len(sentences)}")

    sentence_results = []

    total_merges = 0

    total_preserved = 0

    number_of_batches = (
        len(sentences) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_number, batch_start in enumerate(
        range(0, len(sentences), BATCH_SIZE),
        start=1
    ):

        batch_sentences = sentences[
            batch_start:batch_start + BATCH_SIZE
        ]

        # Tokenize each sentence using the same encoding
        # path as the static XLM-R baseline.
        batch_tokens = [
            tokenize_sentence(tokenizer, text)
            for text in batch_sentences
        ]

        original_batch_tokens = [
            list(tokens)
            for tokens in batch_tokens
        ]

        original_batch_lengths = [
            len(tokens)
            for tokens in original_batch_tokens
        ]

        # Apply Dynamic BPE to this fixed batch.
        merged_batch_tokens, merges_applied = count_merges(
            dynamic_bpe,
            batch_tokens,
            MAX_MERGES
        )

        total_merges += merges_applied

        # Evaluate each sentence independently.
        for index, original_text in enumerate(batch_sentences):

            original_tokens = original_batch_tokens[index]

            merged_tokens = merged_batch_tokens[index]

            original_count = len(original_tokens)

            dynamic_count = len(merged_tokens)

            tokens_saved = (
                original_count - dynamic_count
            )

            if original_count > 0:
                reduction_percent = (
                    tokens_saved / original_count
                ) * 100
            else:
                reduction_percent = 0.0

            decoded_text, preserved = decode_and_check(
                tokenizer,
                original_text,
                merged_tokens
            )

            if preserved:
                total_preserved += 1

            sentence_results.append({
                "language_code": language_code,
                "language": language_name,
                "sentence_id": batch_start + index + 1,
                "original_tokens": original_count,
                "dynamic_tokens": dynamic_count,
                "tokens_saved": tokens_saved,
                "reduction_percent": reduction_percent,
                "text_preserved": preserved,
                "original_text": original_text,
                "decoded_text": decoded_text,
            })

        print(
            f"Batch {batch_number}/{number_of_batches} | "
            f"Sentences: {len(batch_sentences)} | "
            f"Merges: {merges_applied}"
        )

    # -----------------------------------------------------
    # LANGUAGE SUMMARY
    # -----------------------------------------------------

    sentence_count = len(sentence_results)

    mean_original = sum(
        row["original_tokens"]
        for row in sentence_results
    ) / sentence_count

    mean_dynamic = sum(
        row["dynamic_tokens"]
        for row in sentence_results
    ) / sentence_count

    mean_saved = sum(
        row["tokens_saved"]
        for row in sentence_results
    ) / sentence_count

    mean_reduction = sum(
        row["reduction_percent"]
        for row in sentence_results
    ) / sentence_count

    preservation_rate = (
        total_preserved / sentence_count
    ) * 100

    summary = {
        "language_code": language_code,
        "language": language_name,
        "sentence_count": sentence_count,
        "mean_original_tokens": mean_original,
        "mean_dynamic_tokens": mean_dynamic,
        "mean_tokens_saved": mean_saved,
        "mean_reduction_percent": mean_reduction,
        "total_merges": total_merges,
        "preserved_sentences": total_preserved,
        "preservation_rate": preservation_rate,
    }

    print("\nLanguage summary:")
    print(f"Mean original tokens: {mean_original:.4f}")
    print(f"Mean dynamic tokens: {mean_dynamic:.4f}")
    print(f"Mean tokens saved: {mean_saved:.4f}")
    print(f"Mean reduction: {mean_reduction:.2f}%")
    print(f"Total merges: {total_merges}")
    print(
        f"Text preservation: "
        f"{total_preserved}/{sentence_count} "
        f"({preservation_rate:.2f}%)"
    )

    return sentence_results, summary


# ---------------------------------------------------------
# SAVE CSV RESULTS
# ---------------------------------------------------------

def save_results(all_sentence_results, all_summaries):

    sentence_output = (
        OUTPUT_DIR
        / "dynamic_bpe_sentence_results.csv"
    )

    summary_output = (
        OUTPUT_DIR
        / "dynamic_bpe_summary.csv"
    )

    if all_sentence_results:

        with open(
            sentence_output,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    all_sentence_results[0].keys()
                )
            )

            writer.writeheader()

            writer.writerows(all_sentence_results)

    if all_summaries:

        with open(
            summary_output,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(all_summaries[0].keys())
            )

            writer.writeheader()

            writer.writerows(all_summaries)

    print("\n" + "=" * 65)
    print("RESULT FILES SAVED")
    print("=" * 65)

    print(sentence_output)
    print(summary_output)


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------


def main():

    print("=" * 65)
    print("TCMB - DYNAMIC BPE BASELINE EXPERIMENT")
    print("=" * 65)

    # -----------------------------------------------------
    # 1. LOAD ORIGINAL XLM-R TOKENIZER
    # -----------------------------------------------------

    print("\nLoading original XLM-R tokenizer...")

    original_tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH)
    )

    # -----------------------------------------------------
    # 2. CONVERT TOKENIZER TO BYTE-LEVEL FORMAT
    # -----------------------------------------------------

    print("\nConverting tokenizer to byte-level format...")

    tokenizer, added_bytes = convert_to_byte_level(
        original_tokenizer
    )

    print(f"Added byte tokens: {added_bytes}")

    # -----------------------------------------------------
    # 3. INITIALIZE DYNAMIC BPE
    # -----------------------------------------------------

    print("\nInitializing Dynamic_BPE...")

    dynamic_bpe = Dynamic_BPE(
        tokenizer,
        tokenizer_boundary="pretokens"
    )

    print("Dynamic_BPE initialized successfully!")
    print("\nOriginal merge_pair function signature:")
    print(inspect.signature(dynamic_bpe.merge_pair))
    # -----------------------------------------------------
    # 4. INITIALIZE RESULT COLLECTIONS
    # -----------------------------------------------------

    all_sentence_results = []

    all_summaries = []

    # -----------------------------------------------------
    # 5. PROCESS ALL EIGHT LANGUAGES
    # -----------------------------------------------------

    for language_code, language_name in LANGUAGES.items():

        sentence_results, summary = process_language(
            language_code,
            language_name,
            tokenizer,
            dynamic_bpe
        )

        all_sentence_results.extend(sentence_results)

        all_summaries.append(summary)

        # Save after every language so completed results
        # are not lost if a later language fails.

        save_results(
            all_sentence_results,
            all_summaries
        )

    # -----------------------------------------------------
    # 6. DISPLAY FINAL EXPERIMENT SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("ALL LANGUAGES COMPLETED")
    print("=" * 65)

    print("\nFINAL DYNAMIC BPE SUMMARY\n")

    print(
        f"{'Language':<12}"
        f"{'Original':>12}"
        f"{'Dynamic':>12}"
        f"{'Reduction':>14}"
        f"{'Preservation':>16}"
    )

    print("-" * 66)

    for summary in all_summaries:

        print(
            f"{summary['language']:<12}"
            f"{summary['mean_original_tokens']:>12.2f}"
            f"{summary['mean_dynamic_tokens']:>12.2f}"
            f"{summary['mean_reduction_percent']:>13.2f}%"
            f"{summary['preservation_rate']:>15.2f}%"
        )

    print("\nExperiment completed successfully.")

    print("\nOutput files:")
    print(
        OUTPUT_DIR / "dynamic_bpe_sentence_results.csv"
    )
    print(
        OUTPUT_DIR / "dynamic_bpe_summary.csv"
    )

if __name__ == "__main__":
    main()