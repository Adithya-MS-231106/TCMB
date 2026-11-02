from pathlib import Path
import csv
import fasttext
from datasets import load_from_disk


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "flores_plus"
MODEL_PATH = PROJECT_ROOT / "models" / "lid.176.ftz"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "language_id_baseline.csv"


# --------------------------------------------------
# 2. Language mapping
# --------------------------------------------------

LANGUAGES = {
    "eng_Latn": "en",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "kan_Knda": "kn",
    "mal_Mlym": "ml",
    "hin_Deva": "hi",
    "cmn_Hans": "zh",
    "jpn_Jpan": "ja",
}


# --------------------------------------------------
# 3. Load fastText model
# --------------------------------------------------

def load_language_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Language identification model not found: {MODEL_PATH}"
        )

    print("Loading fastText language identification model...")

    return fasttext.load_model(str(MODEL_PATH))


# --------------------------------------------------
# 4. Extract sentence text
# --------------------------------------------------

def extract_text(dataset):

    if "sentence" in dataset.column_names:
        return dataset["sentence"]

    if "text" in dataset.column_names:
        return dataset["text"]

    raise ValueError(
        f"No sentence/text column found. "
        f"Available columns: {dataset.column_names}"
    )


# --------------------------------------------------
# 5. Evaluate language identification
# --------------------------------------------------

def evaluate_language_identification(model):

    results = []

    for language, expected_code in LANGUAGES.items():

        language_dir = DATA_DIR / language

        if not language_dir.exists():
            print(f"Skipping missing language folder: {language}")
            continue

        print(f"\nEvaluating: {language}")

        dataset = load_from_disk(str(language_dir / "devtest"))

        sentences = extract_text(dataset)

        correct = 0
        total = 0

        for sentence in sentences:

            # fastText expects one line of text
            clean_sentence = " ".join(str(sentence).split())

            if not clean_sentence:
                continue

            predictions = model.predict(
                clean_sentence.replace("\n", " "),
                k=1
            )

            predicted_label = predictions[0][0]
            predicted_code = predicted_label.replace("__label__", "")

            is_correct = predicted_code == expected_code

            if is_correct:
                correct += 1

            total += 1

        accuracy = correct / total if total else 0

        results.append({
            "language": language,
            "expected_code": expected_code,
            "total_sentences": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "accuracy_percent": accuracy * 100,
        })

        print(
            f"Accuracy: {accuracy * 100:.2f}% "
            f"({correct}/{total})"
        )

    return results


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

def save_results(results):

    if not results:
        print("No results to save.")
        return

    with open(
        OUTPUT_FILE,
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

    print(f"\nResults saved to: {OUTPUT_FILE}")


# --------------------------------------------------
# 7. Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("TCMB - LANGUAGE IDENTIFICATION BASELINE")
    print("=" * 60)

    model = load_language_model()

    results = evaluate_language_identification(model)

    save_results(results)

    print("\nLanguage identification evaluation completed.")


if __name__ == "__main__":
    main()