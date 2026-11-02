from pathlib import Path
from datasets import load_from_disk

DATA_ROOT = Path("data/flores_plus")

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

for language in LANGUAGES:

    print(f"\nChecking {language}...")

    root_dataset = load_from_disk(
        str(DATA_ROOT / language)
    )

    nested_sentences = root_dataset["devtest"]["text"]

    direct_dataset = load_from_disk(
        str(DATA_ROOT / language / "devtest")
    )

    direct_sentences = direct_dataset["text"]

    print("Root dataset sentences:", len(nested_sentences))
    print("Direct dataset sentences:", len(direct_sentences))

    same_length = len(nested_sentences) == len(direct_sentences)

    same_content = (
        list(nested_sentences) == list(direct_sentences)
    )

    print("Same length:", same_length)
    print("Same sentence content and order:", same_content)

    if not same_content:
        differences = 0

        for i, (a, b) in enumerate(
            zip(nested_sentences, direct_sentences)
        ):
            if a != b:
                differences += 1

                if differences <= 3:
                    print(f"\nMismatch at sentence {i}:")
                    print("Root:", repr(a[:150]))
                    print("Direct:", repr(b[:150]))

        print("Total mismatches:", differences)