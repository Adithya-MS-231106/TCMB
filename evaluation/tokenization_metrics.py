"""Tokenizer-independent fragmentation metrics."""
from typing import Iterable

def _safe_tokens(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        raise ValueError("Pass one string at a time, not a batch.")
    return ids

def tokenization_metrics(tokenizer, text: str):
    ids = _safe_tokens(tokenizer, text)
    n_tokens = len(ids)
    n_chars = len(text)
    words = text.split()
    return {
        "characters": n_chars,
        "whitespace_words": len(words),
        "tokens": n_tokens,
        "tokens_per_whitespace_word": n_tokens / len(words) if words else 0.0,
        "characters_per_token": n_chars / n_tokens if n_tokens else 0.0,
        "tokens_per_100_characters": 100 * n_tokens / n_chars if n_chars else 0.0,
    }

def summarize_tokenization(tokenizer, texts: Iterable[str]):
    rows = [tokenization_metrics(tokenizer, str(text)) for text in texts]
    if not rows:
        raise ValueError("texts cannot be empty.")
    keys = rows[0].keys()
    summary = {"n_sentences": len(rows)}
    for key in keys:
        vals = [row[key] for row in rows]
        summary[f"mean_{key}"] = sum(vals) / len(vals)
    summary["max_tokens"] = max(row["tokens"] for row in rows)
    summary["median_tokens"] = sorted(row["tokens"] for row in rows)[len(rows) // 2]
    return summary
