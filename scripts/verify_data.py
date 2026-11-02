from pathlib import Path
from datasets import load_from_disk

DATASETS = {
    "flores_plus": [
        "eng_Latn", "tam_Taml", "tel_Telu", "kan_Knda",
        "mal_Mlym", "hin_Deva", "cmn_Hans", "jpn_Jpan",
    ],
    "sib200": [
        "eng_Latn", "tam_Taml", "tel_Telu", "kan_Knda",
        "mal_Mlym", "hin_Deva", "zho_Hans", "jpn_Jpan",
    ],
}

for dataset_name, languages in DATASETS.items():
    print(f"\n========== {dataset_name.upper()} ==========")

    for lang in languages:
        path = Path("data") / dataset_name / lang

        if not path.exists():
            print(f"[MISSING] {lang}")
            continue

        try:
            dataset = load_from_disk(str(path))
            print(f"\n[OK] {lang}")

            for split, data in dataset.items():
                print(f"  {split}: {len(data)} examples")
                print(f"  Columns: {data.column_names}")

                if "text" in data.column_names:
                    empty = sum(
                        1 for text in data["text"]
                        if not isinstance(text, str) or not text.strip()
                    )
                    print(f"  Empty/invalid text: {empty}")

        except Exception as error:
            print(f"[ERROR] {lang}: {error}")

print("\nDataset verification completed.")
