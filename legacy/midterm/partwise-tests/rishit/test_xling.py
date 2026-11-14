from dhvani.rank.parser import parse_and_rank
from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex
from dhvani.rank.vsm import query_vector
from dhvani.rank.xling import Translator, load_muse_dict, load_news_dict, translate


def xling_terms(token):
    return {t: w for t, w, s in token["expansions"] if s == "xling"}


def test_news_dict_weights_sum_to_one():
    d = load_news_dict()
    for english, options in d.items():
        assert abs(sum(w for _, w in options) - 1.0) < 1e-9, english


def test_given_weights_are_kept():
    d = load_news_dict()
    assert dict(d["rain"]) == {"बारिश": 0.7, "वर्षा": 0.3}


def test_english_word_gets_hindi_translation():
    q = translate(exact_query("weather"))
    tok = q["tokens"][0]
    assert "मौसम" in xling_terms(tok)
    # The English word stays as an exact expansion too.
    assert ("weather", 1.0, "exact") in tok["expansions"]


def test_weight_is_split_across_translations():
    q = translate(exact_query("rain"))
    w = xling_terms(q["tokens"][0])
    en = q["tokens"][0]["lang"]["en"]
    assert abs(sum(w.values()) - en) < 1e-9
    assert w["बारिश"] > w["वर्षा"]


def test_longest_phrase_wins():
    q = translate(exact_query("prime minister"))
    assert len(q["tokens"]) == 1
    assert q["tokens"][0]["surface"] == "prime minister"
    assert "प्रधानमंत्री" in xling_terms(q["tokens"][0])


def test_multi_word_hindi_translation_is_split():
    q = translate(exact_query("stock market"))
    assert set(xling_terms(q["tokens"][0])) == {"शेयर", "बाजार"}


def test_english_stop_words_are_dropped():
    q = translate(exact_query("the rain in delhi"))
    assert [t["surface"] for t in q["tokens"]] == ["rain", "delhi"]


def test_hindi_tokens_pass_through_unchanged():
    original = exact_query("दिल्ली बारिश")
    assert translate(original)["tokens"] == original["tokens"]


def test_unknown_english_word_is_left_alone():
    q = translate(exact_query("zxqv"))
    assert xling_terms(q["tokens"][0]) == {}


def test_english_query_finds_the_hindi_article():
    idx = SampleIndex.load()
    top = parse_and_rank(translate(exact_query("delhi rain")), idx, k=1)[0]
    assert top[0] == "jagran_1001"
    assert top[2]["stage"] == "all words, with variants"


def test_translations_enter_the_query_vector():
    idx = SampleIndex.load()
    qvec = query_vector(translate(exact_query("weather tomorrow")), idx)
    assert {"मौसम", "कल"} <= set(qvec)
    assert "weather" not in qvec     # not in the Hindi index, so it drops out


def test_without_translation_english_finds_nothing():
    idx = SampleIndex.load()
    assert parse_and_rank(exact_query("weather tomorrow"), idx, k=3) == []


def test_muse_format_is_read_and_news_dict_wins(tmp_path):
    muse = tmp_path / "en-hi.txt"
    muse.write_text("rain बरसात\nrain बारिश\ncar गाड़ी\n", encoding="utf-8")
    assert dict(load_muse_dict(muse)) ["car"] == [("गाड़ी", 1.0)]
    t = Translator(muse_path=muse, learned_path=None)
    assert dict(t.lookup("car")) == {"गाड़ी": 1.0}          # only in MUSE
    assert dict(t.lookup("rain")) == {"बारिश": 0.7, "वर्षा": 0.3}   # news dict wins


def test_missing_muse_file_is_fine(tmp_path):
    assert load_muse_dict(tmp_path / "nope.txt") == {}
