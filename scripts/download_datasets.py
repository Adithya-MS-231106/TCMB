from pathlib import Path
from datasets import load_dataset, load_from_disk

# ==========================================
# TCMB - MULTILINGUAL DATASET DOWNLOADER
# ==========================================

FLORES_LANGUAGES = [
    "eng_Latn", "tam_Taml", "tel_Telu", "kan_Knda",
    "mal_Mlym", "hin_Deva", "cmn_Hans", "jpn_Jpan",
]

SIB200_LANGUAGES = [
    "eng_Latn", "tam_Taml", "tel_Telu", "kan_Knda",
    "mal_Mlym", "hin_Deva", "zho_Hans", "jpn_Jpan",
]

DATASETS = {
    "flores_plus": ("openlanguagedata/flores_plus", FLORES_LANGUAGES),
    "sib200": ("Davlan/sib200", SIB200_LANGUAGES),
}

BASE_DIR = Path("data")


def download_dataset(folder_name, repository, language):
    save_path = BASE_DIR / folder_name / language

    print(f"\n[{folder_name}] Language: {language}")

    # Reuse an already saved dataset
    if (save_path / "dataset_dict.json").exists():
        try:
            existing = load_from_disk(str(save_path))
            print("Already downloaded. Skipping.")
            for split, data in existing.items():
                print(f"  {split}: {len(data)} examples")
            return
        except Exception:
            print("Existing saved dataset could not be loaded.")

    try:
        print("Loading dataset...")

        dataset = load_dataset(repository, language)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        dataset.save_to_disk(str(save_path))

        print("Saved successfully!")

        for split, data in dataset.items():
            print(f"  {split}: {len(data)} examples")

    except Exception as error:
        print(f"FAILED: {language}")
        print(f"Reason: {error}")


def main():
    for folder_name, (repository, languages) in DATASETS.items():
        print(f"\n========== {folder_name.upper()} ==========")
        for language in languages:
            download_dataset(folder_name, repository, language)

if __name__ == "__main__":
    main()
