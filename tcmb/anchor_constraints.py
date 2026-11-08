"""Conservative protection of likely anchors before experimental token merging.

The implementation intentionally errs on the side of caution: it protects
high-confidence spans such as numbers, dates, URLs, emails, and narrow Latin
name-like entities, but it avoids broad "any capitalized word" heuristics that
would incorrectly block ordinary text.
"""
import re
from typing import Iterable


NUMBER_RE = re.compile(
    r"(?<!\w)[+-]?(?:\d+(?:[.,:/-]\d+)*|\d+[.,]\d+|\d+)(?:%)?(?!\w)",
    re.UNICODE,
)
DATE_RE = re.compile(
    r"(?<!\w)(?:\d{1,4}[/-]\d{1,2}[/-]\d{1,4}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})(?!\w)",
    re.UNICODE,
)
LATIN_ENTITY_RE = re.compile(
    r"(?<!\w)(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[A-Z]{2,})(?!\w)",
    re.UNICODE,
)
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>]+", re.UNICODE)
EMAIL_RE = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)",
    re.UNICODE,
)


def _is_name_candidate(match: re.Match[str]) -> bool:
    surface = match.group(0).strip()
    if len(surface) < 2:
        return False
    if surface.lower() in {"a", "i", "an", "the", "and", "or", "to", "of", "on", "in", "for", "with", "at", "by"}:
        return False
    if len(surface.split()) >= 2:
        return any(ch.isalpha() for ch in surface)
    has_multiple_capitals = sum(ch.isupper() for ch in surface) >= 2
    return has_multiple_capitals and len(surface) >= 3


def find_anchor_spans(text: str, extra_terms: Iterable[str] = ()):
    """Return sorted, merged (start, end, kind, surface) spans."""
    if text is None:
        return []

    candidates = []
    patterns = (
        ("number", NUMBER_RE),
        ("date", DATE_RE),
        ("url", URL_RE),
        ("email", EMAIL_RE),
    )

    for kind, pattern in patterns:
        for match in pattern.finditer(str(text)):
            candidates.append((match.start(), match.end(), kind, match.group()))

    for match in LATIN_ENTITY_RE.finditer(str(text)):
        if _is_name_candidate(match):
            candidates.append((match.start(), match.end(), "named_entity_candidate", match.group()))

    for term in extra_terms:
        if not term:
            continue
        start = 0
        while True:
            idx = str(text).find(term, start)
            if idx < 0:
                break
            candidates.append((idx, idx + len(term), "user_term", term))
            start = idx + len(term)

    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    merged = []
    precedence = {"date": 4, "email": 4, "url": 4, "number": 2, "user_term": 3, "named_entity_candidate": 1, "overlap": 0}
    for start, end, kind, surface in candidates:
        if merged and start < merged[-1][1]:
            previous_start, previous_end, previous_kind, _ = merged[-1]
            merged_end = max(previous_end, end)
            merged_surface = str(text)[previous_start:merged_end]
            if precedence.get(kind, 0) >= precedence.get(previous_kind, 0):
                kind_to_keep = kind
            else:
                kind_to_keep = previous_kind
            merged[-1] = (previous_start, merged_end, kind_to_keep, merged_surface)
        else:
            merged.append((start, end, kind, surface))

    return merged


def is_anchor_token(token: str, extra_terms: Iterable[str] = ()):
    stripped = str(token).strip()
    if not stripped:
        return False

    normalized = stripped.strip("\"'`.,;:!?()[]{}<>" )
    if not normalized:
        return False

    extra = {str(value).strip() for value in extra_terms}
    return bool(
        NUMBER_RE.fullmatch(normalized)
        or DATE_RE.fullmatch(normalized)
        or URL_RE.fullmatch(normalized)
        or EMAIL_RE.fullmatch(normalized)
        or normalized in extra
    )
