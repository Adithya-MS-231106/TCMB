from tcmb.anchor_constraints import find_anchor_spans, is_anchor_token
from tcmb.typological_features import sentence_complexity


def test_unicode_punctuation_detection_ignores_combining_marks():
    text = "வணக்கம்́"
    features = sentence_complexity(text)
    assert features["punctuation_ratio"] == 0.0, features
    assert features["mark_ratio"] >= 0.0

    sentence = "This is an English sentence, with punctuation!"
    features = sentence_complexity(sentence)
    assert features["punctuation_ratio"] > 0.0, features


def test_anchor_constraints_detect_high_confidence_spans():
    assert is_anchor_token("2024-11-08")
    assert is_anchor_token("https://example.com")
    assert is_anchor_token("user@example.com")
    assert is_anchor_token("123.45%")

    spans = find_anchor_spans("Contact user@example.com on 2024-11-08 and visit https://example.com.")
    kinds = {kind for _, _, kind, _ in spans}
    assert {"email", "date", "url"}.issubset(kinds)


def test_anchor_constraints_ignore_ordinary_text():
    assert not is_anchor_token("ordinary text")
    assert not is_anchor_token("こんにちは")
    assert not is_anchor_token("hello")
    assert not is_anchor_token("A")

    spans = find_anchor_spans("The quick brown fox jumps over the lazy dog.")
    assert not spans


if __name__ == "__main__":
    test_unicode_punctuation_detection_ignores_combining_marks()
    test_anchor_constraints_detect_high_confidence_spans()
    test_anchor_constraints_ignore_ordinary_text()
    print("Unicode and anchor validation passed.")
