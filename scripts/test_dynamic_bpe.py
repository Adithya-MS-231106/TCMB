from pathlib import Path
from transformers import AutoTokenizer

from baselines.dynamic_bpe import FloresDynamicBPE

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xlm_roberta"

from baselines.dynamic_bpe import (
    REFERENCE_ROOT,
    ZETT_COMPAT,
    ZETT_ORIGINAL,
)

import sys
sys.path.insert(0, str(REFERENCE_ROOT))
sys.path.insert(0, str(ZETT_COMPAT))

import zett
zett.__path__.append(str(ZETT_ORIGINAL))

from zett.tokenizer_converters import convert_to_byte_level

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    use_fast=True,
    local_files_only=True,
)

tokenizer, _ = convert_to_byte_level(tokenizer)

dynamic = FloresDynamicBPE(tokenizer)

sentences = [
    "The language model processes text.",
    "Multilingual tokenization is important for research.",
]

result = dynamic.tokenize_batch(sentences, max_merges=20)

for i, sentence in enumerate(sentences):
    print("\nSentence:", sentence)
    print("Original tokens:", result["original_lengths"][i])
    print("Dynamic tokens:", result["final_lengths"][i])
    print("Merged tokens:", result["tokens"][i])

print("\nTotal merges:", result["total_merges"])
print("TEST COMPLETED")