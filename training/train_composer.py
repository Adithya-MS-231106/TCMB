"""Fit a simple string-level merge composer from a JSONL corpus.

Input: one JSON object per line, e.g. {"tokens": ["New", "York", "is", "large"]}
Output: JSON merge rules. This does not train XLM-R embeddings.
"""
import argparse, json
from pathlib import Path
from tcmb.composer import GreedyMergeComposer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL with a 'tokens' list per line.")
    parser.add_argument("--output", default="results/tcmb_merge_rules.json")
    parser.add_argument("--max-merges", type=int, default=100)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    sequences = []
    with open(args.input, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            tokens = row.get("tokens")
            if not isinstance(tokens, list) or not all(isinstance(x, str) for x in tokens):
                raise ValueError(f"Line {line_no}: expected a string list in 'tokens'.")
            sequences.append(tokens)
    if not sequences:
        raise ValueError("No token sequences found.")
    composer = GreedyMergeComposer().fit(
        sequences, max_merges=args.max_merges, min_frequency=args.min_frequency
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(composer.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(composer.merge_rules)} merge rules to {out}")

if __name__ == "__main__":
    main()
