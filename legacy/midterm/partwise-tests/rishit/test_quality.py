from dhvani.rank.quality import classify, demote, listing_url


class Idx:
    def __init__(self, heads):
        self.articles = {d: {"headline": h} for d, h in heads.items()}
        self.meta = {d: {} for d in heads}


def test_listing_urls():
    assert listing_url("https://www.amarujala.com/technology")
    assert listing_url("https://www.livehindustan.com/bihar/darbhanga/news")
    assert listing_url("https://www.amarujala.com/live/bihar/bihar-latest-and-breaking-news-today")
    assert not listing_url("https://www.jagran.com/delhi/new-delhi-city-earthquake-tremors-40396681.html")
    assert not listing_url("https://www.amarujala.com/dehradun/bjp-councillor-house-robbery-case-news")


def test_classify_headlines():
    assert classify("देवरिया की सबसे ताज़ा खबर") == "listing"
    assert classify("देवरिया की सबसे ताजा खबर") == "listing"          # nukta or not
    assert classify("Guna News, Guna Samachar, गुना समाचार") == "listing"
    assert classify("Meen Tarot Rashifal 7 October 2026: रुटीन रखें") == "horoscope"
    assert classify("भूकंप के झटकों से कांपा उत्तर भारत") == "article"
    assert classify("ताज़ा खबर पर बयान - latest news on the statement") == "article"   # Jagran bilingual article


def test_non_articles_go_after_articles_even_from_a_stricter_stage():
    idx = Idx({"a": "राँची की सबसे ताज़ा खबर", "b": "भूकंप के झटके", "c": "Tula Rashifal आज"})
    results = [("a", 0.9, {"terms": {}, "stage": "all words"}),
               ("c", 0.8, {"terms": {}, "stage": "all words"}),
               ("b", 0.5, {"terms": {}, "stage": "any word"})]
    out = demote(results, idx)
    assert [d for d, _, _ in out] == ["b", "a", "c"]
    assert out[1][2]["page_type"] == "listing" and abs(out[1][1] - 0.27) < 1e-9


def test_articles_keep_their_order():
    idx = Idx({"x": "दिल्ली में बारिश", "y": "कोहली का शतक"})
    results = [("x", 0.9, {"t": 1.0}), ("y", 0.5, {"t": 1.0})]
    assert [d for d, _, _ in demote(results, idx)] == ["x", "y"]
