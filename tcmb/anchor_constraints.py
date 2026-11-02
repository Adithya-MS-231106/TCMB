"""Conservative protection of likely anchors before experimental token merging."""
import re
from typing import Iterable

NUMBER_RE = re.compile(r"\b[+-]?(?:\d+(?:[.,:/-]\d+)*)%?\b")
LATIN_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,})\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")

def find_anchor_spans(text: str, extra_terms: Iterable[str] = ()):
    """Return sorted, merged (start, end, kind, surface) spans."""
    candidates = []
    for kind, pattern in (
        ("number", NUMBER_RE), ("named_entity_candidate", LATIN_ENTITY_RE),
        ("url", URL_RE), ("email", EMAIL_RE)
    ):
        candidates.extend((m.start(), m.end(), kind, m.group()) for m in pattern.finditer(text))
    for term in extra_terms:
        if not term:
            continue
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            candidates.append((idx, idx + len(term), "user_term", term))
            start = idx + len(term)
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    merged = []
    for item in candidates:
        if merged and item[0] < merged[-1][1]:
            old = merged[-1]
            end = max(old[1], item[1])
            surface = text[old[0]:end]
            merged[-1] = (old[0], end, old[2] if old[2] == item[2] else "overlap", surface)
        else:
            merged.append(item)
    return merged

def is_anchor_token(token: str, extra_terms: Iterable[str] = ()):
    stripped = token.strip()
    if not stripped:
        return False
    return bool(
        NUMBER_RE.fullmatch(stripped) or URL_RE.fullmatch(stripped)
        or EMAIL_RE.fullmatch(stripped) or stripped in set(extra_terms)
    )
