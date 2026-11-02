
import sys
import csv
from pathlib import Path
from collections import defaultdict

from datasets import load_from_disk
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --------------------------------------------------
# Reference Dynamic BPE imports
# --------------------------------------------------

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

from tokenizations.dynamic_bpe import Dynamic_BPE
from zett.tokenizer_converters import convert_to_byte_level

# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

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

FLORES_ROOT = ROOT / "data" / "flores_plus"
MODEL_PATH = ROOT / "models" / "xlm_roberta"
TABLES = ROOT / "results" / "tables"

BATCH_SIZE = 32

MATCHED_BUDGET_PATH = (
    TABLES / "dev_matched_budget_batches.csv"
)

BUDGET_CONFIG_PATH = (
    TABLES / "tcmb_budget_config.json"
)

# --------------------------------------------------
# Load and convert tokenizer
# --------------------------------------------------

print("Loading XLM-R tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

tokenizer, added_bytes = convert_to_byte_level(tokenizer)

print(f"Byte-level conversion added {added_bytes} bytes.")

# --------------------------------------------------
# Load matched adaptive budgets
# --------------------------------------------------

with open(
    MATCHED_BUDGET_PATH,
    encoding="utf-8"
) as file:
    matched_rows = list(csv.DictReader(file))

matched_budget_lookup = {}

for row in matched_rows:
    key = (
        row["language"],
        int(row["batch_index"])
    )

    matched_budget_lookup[key] = int(
        row["matched_adaptive_budget"]
    )

# --------------------------------------------------
# Load fixed moderate budgets
# --------------------------------------------------

import json

with open(
    BUDGET_CONFIG_PATH,
    encoding="utf-8"
) as file:
    budget_config = json.load(file)

# --------------------------------------------------
# Dynamic BPE execution
# --------------------------------------------------

def apply_dynamic_bpe(
    tokenizer,
    original_batch_tokens,
    allowed_operations,
):
    """
    Apply the original Dynamic BPE pair-selection
    and merge operations to a fresh copy of a batch.

    actual_operations counts successful merge-rule
    applications, not individual token occurrences.
    """

    merged_tokens = [
        list(tokens)
        for tokens in original_batch_tokens
    ]

    dynamic_bpe = Dynamic_BPE(
        tokenizer,
        tokenizer_boundary="pretokens",
    )

    actual_operations = 0

    for _ in range(allowed_operations):

        pair = dynamic_bpe.get_most_frequent_pair(
            merged_tokens
        )

        if pair is None or pair == "":
            break

        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"Unexpected pair returned: {pair}"
            )

        left, right = pair

        result = dynamic_bpe.merge_pair(
            left,
            right,
            merged_tokens,
            ner=False,
            batch_word_ids=[],
        )

        if result is not None:
            if isinstance(result, tuple):
                merged_tokens = result[0]
            else:
                merged_tokens = result

        actual_operations += 1

    return merged_tokens, actual_operations


# --------------------------------------------------
# Reconstruction and metrics
# --------------------------------------------------

def evaluate_batch(
    original_sentences,
    original_tokens,
    final_tokens,
):
    initial_token_count = sum(
        len(tokens) for tokens in original_tokens
    )

    final_token_count = sum(
        len(tokens) for tokens in final_tokens
    )

    tokens_saved = (
        initial_token_count - final_token_count
    )

    if initial_token_count == 0:
        reduction_percent = 0.0
    else:
        reduction_percent = (
            100.0 * tokens_saved / initial_token_count
        )

    preserved_sentences = 0

    for sentence, tokens in zip(
        original_sentences,
        final_tokens,
    ):
        decoded = tokenizer.convert_tokens_to_string(tokens)

        # Same reconstruction adjustment validated
        # in the earlier Dynamic BPE experiment.
        if decoded.startswith(" "):
            decoded = decoded[1:]

        if decoded == sentence:
            preserved_sentences += 1

    return {
        "initial_tokens": initial_token_count,
        "final_tokens": final_token_count,
        "tokens_saved": tokens_saved,
        "reduction_percent": reduction_percent,
        "preserved_sentences": preserved_sentences,
        "total_sentences": len(original_sentences),
    }


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

batch_results = []
summary_results = []

print("\nFIXED VS ADAPTIVE DYNAMIC BPE")
print("=" * 85)

for language_code, language_name in LANGUAGES.items():

    print(f"\nProcessing {language_name}...")

    dataset = load_from_disk(
        str(FLORES_ROOT / language_code / "dev")
    )

    sentences = dataset["text"]

    fixed_budget = int(
        budget_config["languages"][language_code]
        ["budgets"]["moderate"]
    )

    language_policy_totals = defaultdict(
        lambda: {
            "allowed": 0,
            "actual": 0,
            "initial_tokens": 0,
            "final_tokens": 0,
            "tokens_saved": 0,
            "preserved": 0,
            "sentences": 0,
            "batches": 0,
        }
    )

    for batch_index, start in enumerate(
        range(0, len(sentences), BATCH_SIZE)
    ):

        batch = sentences[start:start + BATCH_SIZE]

        key = (language_code, batch_index)

        if key not in matched_budget_lookup:
            raise KeyError(
                f"Missing matched budget for {key}"
            )

        adaptive_budget = matched_budget_lookup[key]

        # Tokenize once; both policies receive
        # independent copies of identical tokens.
        original_batch_tokens = []

        for sentence in batch:
            encoded = tokenizer(
                sentence,
                add_special_tokens=False,
            )

            tokens = tokenizer.convert_ids_to_tokens(
                encoded["input_ids"]
            )

            original_batch_tokens.append(tokens)

        policies = {
            "fixed": fixed_budget,
            "adaptive": adaptive_budget,
        }

        for policy_name, allowed_budget in policies.items():

            final_tokens, actual_operations = apply_dynamic_bpe(
                tokenizer=tokenizer,
                original_batch_tokens=original_batch_tokens,
                allowed_operations=allowed_budget,
            )

            metrics = evaluate_batch(
                original_sentences=batch,
                original_tokens=original_batch_tokens,
                final_tokens=final_tokens,
            )

            if actual_operations > allowed_budget:
                raise AssertionError(
                    "Actual operations exceeded allowance."
                )

            batch_results.append({
                "language": language_code,
                "language_name": language_name,
                "batch_index": batch_index,
                "policy": policy_name,
                "allowed_operations": allowed_budget,
                "actual_operations": actual_operations,
                **metrics,
            })

            totals = language_policy_totals[policy_name]

            totals["allowed"] += allowed_budget
            totals["actual"] += actual_operations
            totals["initial_tokens"] += metrics["initial_tokens"]
            totals["final_tokens"] += metrics["final_tokens"]
            totals["tokens_saved"] += metrics["tokens_saved"]
            totals["preserved"] += metrics["preserved_sentences"]
            totals["sentences"] += metrics["total_sentences"]
            totals["batches"] += 1

    # --------------------------------------------------
    # Language-level summaries
    # --------------------------------------------------

    for policy_name in ["fixed", "adaptive"]:

        totals = language_policy_totals[policy_name]

        reduction = (
            100.0 * totals["tokens_saved"]
            / totals["initial_tokens"]
            if totals["initial_tokens"] else 0.0
        )

        preservation = (
            100.0 * totals["preserved"]
            / totals["sentences"]
            if totals["sentences"] else 0.0
        )

        summary_results.append({
            "language": language_code,
            "language_name": language_name,
            "policy": policy_name,
            "batches": totals["batches"],
            "allowed_operations": totals["allowed"],
            "actual_operations": totals["actual"],
            "initial_tokens": totals["initial_tokens"],
            "final_tokens": totals["final_tokens"],
            "tokens_saved": totals["tokens_saved"],
            "reduction_percent": reduction,
            "preserved_sentences": totals["preserved"],
            "total_sentences": totals["sentences"],
            "preservation_percent": preservation,
        })

        print(
            f"  {policy_name:8s} "
            f"Allowed={totals['allowed']:6d} "
            f"Actual={totals['actual']:6d} "
            f"Saved={totals['tokens_saved']:6d} "
            f"Reduction={reduction:6.2f}% "
            f"Preservation={preservation:6.2f}%"
        )

    fixed_total = language_policy_totals["fixed"]["allowed"]
    adaptive_total = language_policy_totals["adaptive"]["allowed"]

    assert fixed_total == adaptive_total, (
        f"Budget mismatch for {language_code}"
    )


# --------------------------------------------------
# Save results
# --------------------------------------------------

batch_output = (
    TABLES / "dev_fixed_vs_adaptive_batches.csv"
)

summary_output = (
    TABLES / "dev_fixed_vs_adaptive_summary.csv"
)

with open(
    batch_output,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=batch_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(batch_results)


with open(
    summary_output,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=summary_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary_results)


print("\n" + "=" * 85)
print("EXPERIMENT COMPLETED")
print(f"Batch results: {batch_output}")
print(f"Summary results: {summary_output}")
print(f"Total batch-policy observations: {len(batch_results)}")
print("All total allowed-budget checks passed.")