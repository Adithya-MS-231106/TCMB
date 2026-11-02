"""Leave-one-language-out evaluation split utility."""
import argparse, json
from pathlib import Path

def leave_one_language_out(language_to_examples, held_out_language):
    if held_out_language not in language_to_examples:
        raise KeyError(f"Unknown held-out language: {held_out_language}")
    train = {
        lang: examples for lang, examples in language_to_examples.items()
        if lang != held_out_language
    }
    test = {held_out_language: language_to_examples[held_out_language]}
    return train, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON mapping language codes to example lists.")
    parser.add_argument("--held-out", required=True)
    parser.add_argument("--output-dir", default="results/leave_one_out")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    train, test = leave_one_language_out(data, args.held_out)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "test.json").write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved split with {len(train)} training languages and held-out {args.held_out}.")

if __name__ == "__main__":
    main()
