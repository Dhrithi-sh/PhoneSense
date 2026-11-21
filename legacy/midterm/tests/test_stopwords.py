from text.stopwords import STOPWORDS, is_stopword


def test_stopword_set_contains_expected_terms():
    expected = {
        "के",
        "में",
        "है",
        "की",
        "से",
        "को",
        "कर",
        "और",
        "पर",
        "का",
        "हैं",
        "रह",
        "हो",
        "ने",
        "भी",
    }

    assert STOPWORDS == expected


def test_known_stopwords_are_detected():
    for term in STOPWORDS:
        assert is_stopword(term)


def test_non_stopwords_are_not_detected():
    non_stopwords = [
        "बारिश",
        "मौसम",
        "सरकार",
        "दिल्ली",
        "भारत",
        "प्रदेश",
        "लोग",
    ]

    for term in non_stopwords:
        assert not is_stopword(term)


def test_empty_term_is_not_stopword():
    assert not is_stopword("")


def test_english_term_is_not_stopword():
    assert not is_stopword("rain")


def test_stopword_check_is_exact():
    assert is_stopword("के")
    assert not is_stopword("केला")