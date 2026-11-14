import pytest

from phonesense.index import Analyzer, AnalyzerConfig
from phonesense.index.stopwords import PROTECTED


@pytest.fixture(scope="module")
def an():
    return Analyzer()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("128 GB", ["128gb"]),
        ("128GB", ["128gb"]),
        ("8 gb ram", ["8gb", "ram"]),
        ("5,000 mAh", ["5000mah"]),
        ("5000mAh", ["5000mah"]),
        ('6.1"', ["6.1in"]),
        ("6.1 inch", ["6.1in"]),
        ("6.7 inches", ["6.7in"]),
        ("6.5” display", ["6.5in", "display"]),
        ("48 MP", ["48mp"]),
        ("48 megapixel", ["48mp"]),
        ("120 Hz", ["120hz"]),
        ("1 TB", ["1tb"]),
        ("5 G", ["5g"]),
        ("2 in 1", ["2", "1"]),  # integer + " in " is not inches
    ],
)
def test_unit_joining(an, text, expected):
    assert an.analyze(text) == expected


def test_model_codes_and_decimals_preserved(an):
    assert an.analyze("Galaxy A54 S23 Redmi 13C Nova X5 5G 4G 6.1") == [
        "galaxi",
        "a54",
        "s23",
        "redmi",
        "13c",
        "nova",
        "x5",
        "5g",
        "4g",
        "6.1",
    ]


def test_protected_words_not_removed_or_stemmed(an):
    text = " ".join(sorted(PROTECTED))
    assert an.analyze(text) == sorted(PROTECTED)
    assert an.analyze("the iphone pro max with a mini") == ["iphon", "pro", "max", "mini"]


def test_stopwords_removed_and_positions_keep_gaps(an):
    assert an.analyze_with_positions("phone with a great camera") == [
        ("phone", 0),
        ("great", 3),
        ("camera", 4),
    ]
    assert an.analyze("the phone's camera isn't bad") == ["phone", "camera", "bad"]


def test_digit_tokens_not_stemmed():
    porter = Analyzer()
    assert porter.analyze("batteries 2023s 5000mahs a54s") == [
        "batteri",
        "2023s",
        "5000mahs",
        "a54s",
    ]
    assert Analyzer(stemmer="light").analyze("batteries phones") == ["battery", "phone"]
    assert Analyzer(stem=False).analyze("batteries phones") == ["batteries", "phones"]


def test_nfkc_and_case(an):
    # full-width digits/letters and ligatures fold to ASCII under NFKC
    assert an.analyze("１２８ＧＢ ﬁngerprint") == ["128gb", "fingerprint"]
    assert an.analyze("SAMSUNG") == an.analyze("samsung")


def test_documents_and_queries_analysed_identically(an, make_phone):
    from phonesense.index import InvertedIndex

    idx = InvertedIndex.build(
        [make_phone("D1", "Samsung Galaxy A54 5G 128GB", ["5,000mAh battery", '6.4" screen'])], an
    )
    for query in ("128 gb", "5000 mah", "6.4 inch", "Galaxy", "A54"):
        terms = an.analyze(query)
        assert terms and all(idx.df(t) == 1 for t in terms), query


def test_config_hash_changes_with_config():
    hashes = {
        Analyzer(**kw).config_hash()
        for kw in ({}, {"stem": False}, {"stopwords": False}, {"stemmer": "light"})
    }
    assert len(hashes) == 4
    assert Analyzer().config_hash() == Analyzer(AnalyzerConfig()).config_hash()


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        AnalyzerConfig(stemmer="snowball")
    with pytest.raises(ValueError):
        AnalyzerConfig.from_dict({"lowercase": True})


def test_from_config_matches_yaml():
    from phonesense.config import get_config

    assert Analyzer.from_config().config == AnalyzerConfig.from_dict(get_config()["analyzer"])
