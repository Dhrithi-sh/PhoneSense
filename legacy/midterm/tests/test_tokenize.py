from text.tokenize import tokenize


def test_hindi_sentence():
    text = "प्रधानमंत्री ने दिल्ली में १२३ लोगों से बात की।"

    assert tokenize(text) == [
        "प्रधानमंत्री",
        "ने",
        "दिल्ली",
        "में",
        "१२३",
        "लोगों",
        "से",
        "बात",
        "की",
    ]


def test_hinglish_sentence():
    text = "Modi ने कहा कि India आगे बढ़ रहा है"

    assert tokenize(text) == [
        "Modi",
        "ने",
        "कहा",
        "कि",
        "India",
        "आगे",
        "बढ़",
        "रहा",
        "है",
    ]


def test_punctuation_is_removed():
    text = "नमस्ते, दुनिया! कैसे हो?"

    assert tokenize(text) == [
        "नमस्ते",
        "दुनिया",
        "कैसे",
        "हो",
    ]


def test_devanagari_digits():
    text = "भारत में १२३ लोग हैं।"

    assert tokenize(text) == [
        "भारत",
        "में",
        "१२३",
        "लोग",
        "हैं",
    ]


def test_english_digits():
    text = "There are 123 people."

    assert tokenize(text) == [
        "There",
        "are",
        "123",
        "people",
    ]


def test_hyphenated_words_are_split():
    text = "AI-based technology"

    assert tokenize(text) == [
        "AI",
        "based",
        "technology",
    ]


def test_empty_text():
    assert tokenize("") == []


def test_only_punctuation():
    assert tokenize("।,!?;:") == []