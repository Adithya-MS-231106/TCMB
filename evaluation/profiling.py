"""Simple, reproducible runtime profiling helpers."""
import time
import statistics
from typing import Callable, Any

def profile_callable(fn: Callable, *args, repeats: int = 10, warmup: int = 2, **kwargs):
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup must be >= 0.")
    for _ in range(warmup):
        fn(*args, **kwargs)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        durations.append(time.perf_counter() - start)
    total = sum(durations)
    return {
        "repeats": repeats,
        "mean_seconds": statistics.mean(durations),
        "median_seconds": statistics.median(durations),
        "stdev_seconds": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "throughput_calls_per_second": repeats / total if total else float("inf"),
    }

def profile_tokenizer(tokenizer, texts, repeats: int = 3, **tokenizer_kwargs):
    texts = list(texts)
    if not texts:
        raise ValueError("texts cannot be empty.")
    def run():
        return tokenizer(texts, **tokenizer_kwargs)
    timing = profile_callable(run, repeats=repeats, warmup=1)
    encoded = tokenizer(texts, **tokenizer_kwargs)
    lengths = [len(x) for x in encoded["input_ids"]]
    timing.update({
        "n_texts": len(texts),
        "total_tokens": sum(lengths),
        "mean_tokens_per_text": sum(lengths) / len(lengths),
        "tokens_per_second": sum(lengths) / timing["mean_seconds"] if timing["mean_seconds"] else 0.0,
    })
    return timing
