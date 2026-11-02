from pathlib import Path
from datasets import load_from_disk

languages = [
    "eng_Latn",
    "tam_Taml",
    "tel_Telu",
    "kan_Knda",
    "mal_Mlym",
    "hin_Deva",
    "cmn_Hans",
    "jpn_Jpan",
]

reference_ids = None
all_aligned = True

for language in languages:
    path = Path("data/flores_plus") / language
    dataset = load_from_disk(str(path))
    ids = dataset["devtest"]["id"]

    if reference_ids is None:
        reference_ids = ids
        print(f"Reference language: {language}")
        print(f"Reference sentence count: {len(ids)}")
        continue

    aligned = ids == reference_ids

    print(
        f"{language}: "
        f"{'ALIGNED' if aligned else 'NOT ALIGNED'}"
    )

    if not aligned:
        all_aligned = False

print("\nAlignment verification:")
print(
    "All languages are aligned."
    if all_aligned
    else "Some languages are not aligned. Investigate before comparison."
)
