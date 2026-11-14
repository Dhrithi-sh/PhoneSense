from text.stem import stem


def test_common_noun_inflections():
    assert stem("लड़की") == "लड़क"
    assert stem("लड़कियों") == "लड़क"
    assert stem("किताबों") == "किताब"


def test_verb_inflections():
    assert stem("खाना") == "खा"
    assert stem("खाता") == "खा"
    assert stem("खाती") == "खा"


def test_word_without_matching_suffix():
    assert stem("भारत") == "भारत"


def test_short_words_are_protected():
    assert stem("खा") == "खा"
    assert stem("का") == "का"


def test_longest_matching_suffix():
    assert stem("लड़कियों") == "लड़क"


def test_empty_word():
    assert stem("") == ""