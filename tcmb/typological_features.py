"""Transparent, manually specified typological metadata for TCMB experiments.

Values are coarse research annotations, not authoritative linguistic claims.
Replace/refine them with documented WALS/URIEL features before publication.
"""
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

def get_profile(language: str):
    if language not in PROFILES:
        raise KeyError(f"No typological profile for {language!r}. Add it to PROFILES.")
    return PROFILES[language]

def profile_as_dict(language: str):
    return asdict(get_profile(language))

def sentence_complexity(text: str):
    """Small observable complexity feature vector; not a linguistic parser."""
    text = str(text)
    return {
        "char_count": len(text),
        "whitespace_word_count": len(text.split()),
        "digit_ratio": sum(ch.isdigit() for ch in text) / max(len(text), 1),
        "punctuation_ratio": sum(not ch.isalnum() and not ch.isspace() for ch in text) / max(len(text), 1),
        "whitespace_ratio": sum(ch.isspace() for ch in text) / max(len(text), 1),
    }
