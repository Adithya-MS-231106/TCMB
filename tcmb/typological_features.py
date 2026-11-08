import unicodedata
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class TypologicalProfile:
    language: str
    script: str
    morphology: str
    whitespace_word_boundaries: bool
    alphabetic: bool
    notes: str = ""

PROFILES = {
    "eng_Latn": TypologicalProfile("eng_Latn", "Latin", "moderately_inflectional", True, True, "English"),
    "tam_Taml": TypologicalProfile("tam_Taml", "Tamil", "agglutinative", True, True, "Tamil"),
    "tel_Telu": TypologicalProfile("tel_Telu", "Telugu", "agglutinative", True, True, "Telugu"),
    "kan_Knda": TypologicalProfile("kan_Knda", "Kannada", "agglutinative", True, True, "Kannada"),
    "mal_Mlym": TypologicalProfile("mal_Mlym", "Malayalam", "agglutinative", True, True, "Malayalam"),
    "hin_Deva": TypologicalProfile("hin_Deva", "Devanagari", "inflectional", True, True, "Hindi"),
    "cmn_Hans": TypologicalProfile("cmn_Hans", "Han", "isolating", False, False, "Simplified Chinese"),
    "jpn_Jpan": TypologicalProfile("jpn_Jpan", "Japanese", "agglutinative", False, False, "Japanese"),
}
def _is_unicode_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P")

def _is_word_char(char: str) -> bool:
    return char.isalpha() or unicodedata.category(char).startswith("M")

def get_profile(language: str):
    if language not in PROFILES:
        raise KeyError(f"No typological profile for {language!r}. Add it to PROFILES.")
    return PROFILES[language]

def profile_as_dict(language: str):
    return asdict(get_profile(language))

def sentence_complexity(text: str):
    """Small observable complexity feature vector; not a linguistic parser."""
    text = str(text)
    total_chars = max(len(text), 1)
    alphabetic_count = sum(_is_word_char(ch) for ch in text)
    digit_count = sum(ch.isdigit() for ch in text)
    whitespace_count = sum(ch.isspace() for ch in text)
    punctuation_count = sum(_is_unicode_punctuation(ch) for ch in text)
    mark_count = sum(unicodedata.category(ch).startswith("M") for ch in text)
    return {
        "char_count": len(text),
        "whitespace_word_count": len(text.split()),
        "alphabetic_count": alphabetic_count,
        "alphabetic_ratio": alphabetic_count / total_chars,
        "digit_count": digit_count,
        "digit_ratio": digit_count / total_chars,
        "punctuation_count": punctuation_count,
        "punctuation_ratio": punctuation_count / total_chars,
        "whitespace_count": whitespace_count,
        "whitespace_ratio": whitespace_count / total_chars,
        "mark_count": mark_count,
        "mark_ratio": mark_count / total_chars,
    }