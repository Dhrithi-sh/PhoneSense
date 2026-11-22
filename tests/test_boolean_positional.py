import importlib.util
import random
from pathlib import Path

import pytest

from phonesense.index import Analyzer, InvertedIndex
from phonesense.retrieval.bm25 import BM25Retriever
from phonesense.retrieval.boolean import BooleanSearcher, Stats, intersect_sorted, parse
from phonesense.retrieval.positional import (
    PhraseBoostRetriever,
    parse_positional,
    phrase_matches,
    proximity_matches,
    query_phrases,
)
from phonesense.schemas import Query

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("seed", range(20))
def test_skip_intersection_equals_naive(seed):
    rng = random.Random(seed)
    a = sorted(rng.sample(range(5000), rng.randint(0, 800)))
    b = sorted(rng.sample(range(5000), rng.randint(0, 50)))
    expected = sorted(set(a) & set(b))
    s1, s2 = Stats(), Stats()
    assert intersect_sorted(a, b, True, s1) == expected
    assert intersect_sorted(b, a, True) == expected
    assert intersect_sorted(a, b, False, s2) == expected
    if len(a) > 400 and len(b) > 5:
        assert s1.comparisons <= s2.comparisons


@pytest.fixture
def idx(make_phone):
    # stem=False so expected terms are literal; positions keep stopword gaps
    docs = [
        make_phone("D0", "samsung galaxy 5g", ["wireless charging and fast charging"]),
        make_phone("D1", "samsung galaxy refurbished", ["charging wireless pad"]),
        make_phone("D2", "pixel 5g", ["battery life is great", "long battery and good life"]),
        make_phone("D3", "iphone 5g renewed", ["wireless fast charging"]),
        make_phone("D4", "nokia classic", ["battery battery life life"]),
    ]
    return InvertedIndex.build(docs, Analyzer(stem=False))


def test_boolean_and_or_not(idx):
    bs = BooleanSearcher(idx)
    assert bs.search("samsung AND 5g") == ["D0"]
    assert bs.search("samsung AND 5g AND NOT refurbished") == ["D0"]
    assert bs.search("samsung AND NOT 5g") == ["D1"]
    assert bs.search("pixel OR iphone") == ["D2", "D3"]
    assert bs.search("NOT 5g") == ["D1", "D4"]
    assert bs.search("samsung 5g") == ["D0"]  # implicit AND


def test_boolean_precedence_and_parentheses(idx):
    bs = BooleanSearcher(idx)
    # left to right: (pixel OR samsung) AND refurbished
    assert bs.search("pixel OR samsung AND refurbished") == ["D1"]
    assert bs.search("pixel OR (samsung AND refurbished)") == ["D1", "D2"]
    assert bs.search("NOT (5g OR nokia)") == ["D1"]
    assert bs.search("galaxy", field="title") == ["D0", "D1"]
    assert bs.search("Wireless AND charging") == ["D0", "D1", "D3"]  # analyzer lowercases
    assert bs.search("zzz OR nokia") == ["D4"]
    for bad in ("AND samsung", "(samsung", "samsung )", ""):
        with pytest.raises(ValueError):
            parse(bad)


def test_boolean_skips_flag_same_results(idx):
    assert BooleanSearcher(idx, skips=False).search("charging AND 5g") == BooleanSearcher(
        idx
    ).search("charging AND 5g")


def test_phrase_order_matters(idx):
    # D0 features text: wireless(3) charging(4) [and] fast(6) charging(7)
    m = phrase_matches(idx, "wireless charging")
    assert m == {0: 1}  # D1 has "charging wireless", D3 has "wireless fast charging"
    assert phrase_matches(idx, "fast charging") == {0: 1, 3: 1}
    assert phrase_matches(idx, "charging wireless") == {1: 1}
    assert phrase_matches(idx, "battery life") == {2: 1, 4: 1}
    assert phrase_matches(idx, "life battery") == {}
    assert phrase_matches(idx, "the") == {}
    assert phrase_matches(idx, "galaxy", field="title") == {0: 1, 1: 1}


def test_phrase_with_stopword_gap(make_phone):
    idx = InvertedIndex.build(
        [make_phone("A", "phone with camera"), make_phone("B", "phone camera")],
        Analyzer(stem=False),
    )
    assert phrase_matches(idx, "phone with camera") == {0: 1}
    assert phrase_matches(idx, "phone camera") == {1: 1}


def test_proximity_window_edges(idx):
    # D2 text: pixel0 5g1 battery2 life3 [is] great5 long6 battery7 [and] good9 life10
    # battery {2, 7}, life {3, 10}; pairs and distances: (2,3)=1, (7,10)=3, (7,3)=4, (2,10)=8
    assert proximity_matches(idx, "battery", "life", 1)[2] == 1
    assert proximity_matches(idx, "battery", "life", 2)[2] == 1
    assert proximity_matches(idx, "battery", "life", 3)[2] == 2  # k = 3 boundary included
    assert proximity_matches(idx, "life", "battery", 4)[2] == 3  # either order: life3-battery7
    assert proximity_matches(idx, "battery", "life", 8)[2] == 4
    assert proximity_matches(idx, "battery", "life", 0) == {}
    with pytest.raises(ValueError):
        proximity_matches(idx, "battery", "life", -1)


def test_parse_positional_and_query_phrases():
    phrases, prox, rest = parse_positional('"wireless charging" battery /3 life samsung')
    assert phrases == ["wireless charging"] and prox == [("battery", "life", 3)]
    assert rest == "samsung"
    assert query_phrases("phone with Fast Charging", ["fast charging", "battery life"]) == [
        "fast charging"
    ]
    assert query_phrases('"night mode" camera', ["night mode"]) == ["night mode"]


def test_phrase_boost_zero_equals_bm25(idx):
    base = BM25Retriever(index=idx)
    zero = PhraseBoostRetriever(base, 0.0, auto_phrases=["fast charging"])
    for text in ("fast charging phone", '"wireless charging" samsung', "battery life"):
        q = Query("q01", text, "dev")
        assert zero.search(q) == base.search(Query("q01", text.replace('"', ""), "dev"))
    assert zero.name.endswith("+phrase")


def test_phrase_boost_promotes_exact_phrase(idx):
    base = BM25Retriever(index=idx)
    boosted = PhraseBoostRetriever(base, 2.0, auto_phrases=[])
    q = Query("q01", '"wireless charging"', "dev")
    plain = dict(base.search(Query("q01", "wireless charging", "dev")))
    got = dict(boosted.search(q))
    import math

    assert got["D0"] == pytest.approx(plain["D0"] + 2.0 * math.log1p(1))
    assert got["D1"] == pytest.approx(plain["D1"])  # reversed order: no boost
    assert boosted.search(q)[0][0] == "D0"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("split", ["test", "all"])
def test_tune_phrase_refuses_non_dev(split, capsys):
    assert _load("tune_phrase").main(["--split", split]) == 2
    assert "only tunes on the dev queries" in capsys.readouterr().err
