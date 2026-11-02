from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

DATA_ROOT = Path("data/flores_plus")
MODEL_PATH = Path("models/xlm_roberta")

LANGUAGES = [
    "eng_Latn", "tam_Taml", "tel_Telu", "kan_Knda",
    "mal_Mlym", "hin_Deva", "cmn_Hans", "jpn_Jpan",
]

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

for language in LANGUAGES:

    dataset = load_from_disk(
        str(DATA_ROOT / language / "devtest")
    )

    raw_counts = []
    stripped_counts = []
    whitespace_sentences = 0
    changed_counts = 0

    for text in dataset["text"]:

        raw_text = str(text)
        stripped_text = raw_text.strip()

        raw_count = len(
            tokenizer(
                raw_text,
                add_special_tokens=False
            )["input_ids"]
        )

        stripped_count = len(
            tokenizer(
                stripped_text,
                add_special_tokens=False
            )["input_ids"]
        )

        raw_counts.append(raw_count)
        stripped_counts.append(stripped_count)

        if raw_text != stripped_text:
            whitespace_sentences += 1

        if raw_count != stripped_count:
            changed_counts += 1

    print(f"\n{language}")
    print(f"Raw mean: {sum(raw_counts) / len(raw_counts):.4f}")
    print(
        f"Stripped mean: "
        f"{sum(stripped_counts) / len(stripped_counts):.4f}"
    )
    print(f"Sentences with surrounding whitespace: {whitespace_sentences}")
    print(f"Sentences with changed token counts: {changed_counts}")