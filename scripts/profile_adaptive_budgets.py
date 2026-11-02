
import sys
import csv
from pathlib import Path
from statistics import mean

from datasets import load_from_disk
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcmb.budget_allocator import FragmentationBudgetAllocator


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

BATCH_SIZE = 32

allocator = FragmentationBudgetAllocator(
    budget_config_path=(
        ROOT / "results" / "tables" / "tcmb_budget_config.json"
    ),
    feature_config_path=(
        ROOT / "results" / "tables" / "tcmb_batch_feature_config.json"
    ),
    budget_level="moderate",
    adaptation_strength=1.0,
)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

# Use the same byte-level conversion as feature calibration
reference_root = Path(
    r"C:\Users\vindh\AI-ML Projects\dynamic-tokenization-reference"
)

compat_dir = reference_root / "zett-source" / "zett-compat"
original_zett_dir = (
    reference_root / "zett-source" / "zett-python-only" / "zett"
)

sys.path.insert(0, str(reference_root))
sys.path.insert(0, str(compat_dir))

import zett
zett.__path__.append(str(original_zett_dir))

from zett.tokenizer_converters import convert_to_byte_level

tokenizer, added_bytes = convert_to_byte_level(tokenizer)

print(
    f"Byte-level conversion added {added_bytes} bytes."
)

output_dir = ROOT / "results" / "tables"
output_dir.mkdir(parents=True, exist_ok=True)

batch_rows = []
summary_rows = []

print("TCMB ADAPTIVE BUDGET DEVELOPMENT PROFILE")
print("=" * 75)

for language_code, language_name in LANGUAGES.items():

    dataset_path = FLORES_ROOT / language_code / "dev"
    dataset = load_from_disk(str(dataset_path))
    sentences = dataset["text"]

    language_budgets = []

    for batch_index, start in enumerate(
        range(0, len(sentences), BATCH_SIZE)
    ):
        batch = sentences[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )

        total_tokens = sum(
            len(ids) for ids in encoded["input_ids"]
        )

        total_chars = sum(
            len("".join(text.split())) for text in batch
        )

        if total_chars == 0:
            raise ValueError(
                f"Empty character batch: {language_code}, "
                f"batch {batch_index}"
            )

        token_density = (
            100.0 * total_tokens / total_chars
        )

        prediction = allocator.predict(
            language=language_code,
            token_density=token_density,
        )

        budget = prediction["predicted_budget"]
        language_budgets.append(budget)

        batch_rows.append({
            "language": language_code,
            "language_name": language_name,
            "batch_index": batch_index,
            "sentence_count": len(batch),
            "token_count": total_tokens,
            "character_count": total_chars,
            "token_density": token_density,
            "density_z_score": prediction["density_z_score"],
            "base_budget": prediction["base_budget"],
            "adaptive_budget": budget,
            "minimum_budget": prediction["minimum_budget"],
            "maximum_budget": prediction["maximum_budget"],
        })

    summary_rows.append({
        "language": language_code,
        "language_name": language_name,
        "batch_count": len(language_budgets),
        "mean_adaptive_budget": mean(language_budgets),
        "min_adaptive_budget": min(language_budgets),
        "max_adaptive_budget": max(language_budgets),
        "mean_base_budget": language_budgets[0] * 0
            + allocator.predict(
                language_code,
                allocator.feature_config["languages"][language_code]
                ["metrics"]["token_density"]["mean"]
            )["base_budget"],
        "total_adaptive_budget": sum(language_budgets),
    })

    print(
        f"{language_name:10s} "
        f"Batches={len(language_budgets):2d} "
        f"Mean={mean(language_budgets):7.2f} "
        f"Min={min(language_budgets):4d} "
        f"Max={max(language_budgets):4d} "
        f"Total={sum(language_budgets):5d}"
    )


batch_path = output_dir / "dev_adaptive_budget_batches.csv"
summary_path = output_dir / "dev_adaptive_budget_summary.csv"

with open(batch_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=batch_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(batch_rows)

with open(summary_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=summary_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("\nSaved:")
print(batch_path)
print(summary_path)
print("\nDevelopment profiling completed.")
