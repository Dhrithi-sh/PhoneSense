import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text.normalize import normalize


def test_nfc_normalization():
    text = "हिन्दी"
    assert normalize(text) == "हिन्दी"


def test_hindi_spelling_variant():
    assert normalize("हिंदी") == "हिन्दी"
    assert normalize("हिन्दी") == "हिन्दी"


def test_zero_width_characters():
    text = "हि\u200cन्दी"
    assert normalize(text) == "हिन्दी"


def test_whitespace():
    text = "  भारत   एक   देश  "
    assert normalize(text) == "भारत एक देश"


def test_devanagari_digits_are_preserved():
    text = "भारत में १२३ लोग हैं"
    assert normalize(text) == text

def test_normalize_lowercases_english_text():
    assert normalize("Delhi DELHI delhi") == "delhi delhi delhi"


def test_normal_words_are_not_corrupted():
    words = [
        "संगीत",
        "अंग्रेज़ी",
        "मंगल",
        "गंगा",
    ]

    for word in words:
        assert normalize(word) == word