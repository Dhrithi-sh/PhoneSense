from dhvani.query import langid


def test_script_detection():
    assert langid.script("मौसम") == "devanagari"
    assert langid.script("mausam") == "roman"
    assert langid.script("kal123") == "roman"


def test_devanagari_is_certainly_hindi():
    assert langid.classify("मौसम") == {"hi": 1.0, "hinglish": 0.0, "en": 0.0}


def test_romanised_hindi_leans_hinglish():
    lang = langid.classify("mausam")
    assert lang["hinglish"] == 0.9
    assert lang["en"] == 0.1
    assert lang["hi"] == 0.0


def test_known_english_leans_english():
    lang = langid.classify("weather")
    assert lang["en"] == 0.9
    assert lang["hinglish"] == 0.1


def test_ambiguous_words_keep_both_readings():
    # "main" is both English and मैं; neither reading is forced to zero.
    for word in ("main", "to", "hi"):
        lang = langid.classify(word)
        assert lang["hinglish"] == 0.5
        assert lang["en"] == 0.5


def test_real_english_words_are_detected_not_just_the_seed():
    # These are plain English, not in any hand-written seed, but must read English
    # (the bundled wordlist) so they don't get phonetic junk.
    for word in ("farmers", "snow", "earthquake", "worried", "flooded"):
        lang = langid.classify(word)
        assert lang["en"] == 0.9, f"{word} should read English"


def test_common_hindi_words_are_protected_from_the_english_list():
    # "kal", "ka", "hai" are English words too, but as Hindi they must stay Hinglish.
    for word in ("kal", "ka", "hai", "mausam", "chai"):
        lang = langid.classify(word)
        assert lang["hinglish"] == 0.9, f"{word} should stay Hinglish"


def test_weights_sum_to_one():
    for word in ("मौसम", "mausam", "weather", "main"):
        assert abs(sum(langid.classify(word).values()) - 1.0) < 1e-9
