from pathlib import Path
import sys

# Reference implementation paths
REFERENCE_ROOT = Path(
    r"C:\Users\vindh\AI-ML Projects\dynamic-tokenization-reference"
)

ZETT_COMPAT = REFERENCE_ROOT / "zett-source" / "zett-compat"
ZETT_ORIGINAL = (
    REFERENCE_ROOT / "zett-source" / "zett-python-only" / "zett"
)

sys.path.insert(0, str(REFERENCE_ROOT))
sys.path.insert(0, str(ZETT_COMPAT))

import zett

if str(ZETT_ORIGINAL) not in zett.__path__:
    zett.__path__.append(str(ZETT_ORIGINAL))

from tokenizations.dynamic_bpe import Dynamic_BPE


class FloresDynamicBPE:
    """FLORES+ adapter around the reference Dynamic_BPE merge algorithm."""

    def __init__(self, tokenizer, boundary="pretokens"):
        self.tokenizer = tokenizer
        self.dynamic_bpe = Dynamic_BPE(
            tokenizer,
            tokenizer_boundary=boundary,
        )

    def tokenize_batch(self, sentences, max_merges=1000):
        if not sentences:
            raise ValueError("Sentence batch cannot be empty.")

        # Tokenize each sentence using the converted base tokenizer.
        batch_tokens = [
            self.tokenizer.tokenize(sentence)
            for sentence in sentences
        ]

        original_lengths = [len(tokens) for tokens in batch_tokens]

        total_merges = 0

        # Apply the reference algorithm's batch-level merge policy.
        while total_merges < max_merges:
            best_pair = self.dynamic_bpe.get_most_frequent_pair(
                batch_tokens=batch_tokens
            )

            if best_pair == "":
                break

            batch_tokens, _ = self.dynamic_bpe.merge_pair(
                a=best_pair[0],
                b=best_pair[1],
                batch_tokens=batch_tokens,
                ner=False,
            )

            total_merges += 1

        final_lengths = [len(tokens) for tokens in batch_tokens]

        return {
            "tokens": batch_tokens,
            "original_lengths": original_lengths,
            "final_lengths": final_lengths,
            "total_merges": total_merges,
        }