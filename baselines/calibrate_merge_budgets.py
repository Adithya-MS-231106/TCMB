
from pathlib import Path
import csv
import json
import math
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "results" / "tables"
    / "dev_merge_budget_batches.csv"
)

OUTPUT_FILE = (
    ROOT / "results" / "tables"
    / "tcmb_budget_config.json"
)

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

BUDGET_LEVELS = {
    "conservative": 0.25,
    "moderate": 0.50,
    "aggressive": 0.75,
}


def calculate_median(values):
    values = sorted(values)
    n = len(values)

    middle = n // 2

    if n % 2 == 1:
        return float(values[middle])

    return (values[middle - 1] + values[middle]) / 2


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input profiling file not found: {INPUT_FILE}"
        )

    language_merges = {
        language: [] for language in LANGUAGES
    }

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "language",
            "merge_operations",
            "hit_cap",
        }

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing columns in profiling CSV: {missing}"
            )

        for row in reader:

            language = row["language"]

            if language not in language_merges:
                raise ValueError(
                    f"Unexpected language in CSV: {language}"
                )

            if row["hit_cap"].strip().lower() == "true":
                raise ValueError(
                    f"{language} has a batch that hit the merge cap. "
                    "Re-profile before calibrating."
                )

            language_merges[language].append(
                int(row["merge_operations"])
            )

    config = {
        "experiment": "TCMB merge-budget calibration",
        "calibration_split": "FLORES+ dev",
        "evaluation_split": "FLORES+ devtest",
        "batch_size": 32,
        "calibration_method": (
            "Median feasible merge operations per language, "
            "multiplied by the specified budget fraction."
        ),
        "budget_levels": {},
        "languages": {},
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    for level, fraction in BUDGET_LEVELS.items():
        config["budget_levels"][level] = fraction

    print("\nTCMB BUDGET CALIBRATION")
    print("=" * 65)

    for language in LANGUAGES:

        values = language_merges[language]

        if len(values) != 32:
            raise ValueError(
                f"{language}: expected 32 batches, "
                f"found {len(values)}."
            )

        median_merges = calculate_median(values)

        budgets = {
            level: max(
                1,
                math.floor(median_merges * fraction)
            )
            for level, fraction in BUDGET_LEVELS.items()
        }

        config["languages"][language] = {
            "profiled_batches": len(values),
            "median_feasible_merges": median_merges,
            "min_feasible_merges": min(values),
            "max_feasible_merges": max(values),
            "budgets": budgets,
        }

        print(
            f"{language:10s} "
            f"median={median_merges:6.1f} "
            f"conservative={budgets['conservative']:4d} "
            f"moderate={budgets['moderate']:4d} "
            f"aggressive={budgets['aggressive']:4d}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nCalibration saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
