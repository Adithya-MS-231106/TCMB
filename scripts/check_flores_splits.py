
from pathlib import Path
from datasets import load_from_disk

root = Path("data/flores_plus")

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

for language in languages:
    dev_path = root / language / "dev"
    devtest_path = root / language / "devtest"

    if not dev_path.exists() or not devtest_path.exists():
        print(f"{language}: MISSING SPLIT")
        continue

    dev = load_from_disk(str(dev_path))
    devtest = load_from_disk(str(devtest_path))

    print(
        f"{language}: "
        f"dev={len(dev)}, "
        f"devtest={len(devtest)}"
    )
