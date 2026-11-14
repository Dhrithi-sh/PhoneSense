import pytest

from text.aggressive_stem import stem_aggressive


@pytest.mark.parametrize(
    "word, expected",
    [
        ("लड़कियाँ", "लड़क"),
        ("लड़कियों", "लड़क"),
        ("किताबों", "किताब"),
        ("लड़कों", "लड़क"),
        ("लड़के", "लड़क"),
        ("खाना", "खान"),
        ("खाती", "खात"),
        ("खाते", "खात"),
        ("भारत", "भारत"),
        ("दिल्ली", "दिल्ल"),
    ],
)
def test_aggressive_stemming(word, expected):
    assert stem_aggressive(word) == expected


def test_empty_word():
    assert stem_aggressive("") == ""


def test_short_word_is_protected():
    assert stem_aggressive("का") == "का"


def test_word_without_suffix_is_unchanged():
    assert stem_aggressive("मौसम") == "मौसम"