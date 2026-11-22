from dhvani.query.roman import romanize


def test_final_schwa_is_dropped():
    # The headline rule from the plan: कमल -> kamal, not kamala.
    assert romanize("कमल") == "kamal"
    assert romanize("कल") == "kal"


def test_matras_and_schwa_drop_together():
    # म + ौ = mau, स = sa, म = ma -> drop final a -> mausam.
    assert romanize("मौसम") == "mausam"
    # ब + ा = baa, र + ि = ri, श = sha -> baarish.
    assert romanize("बारिश") == "baarish"


def test_virama_removes_inherent_vowel():
    # द + ि = di, ल + ् = l (virama kills the vowel), ल + ी = lii.
    assert romanize("दिल्ली") == "dillii"


def test_word_ending_in_a_matra_keeps_its_vowel():
    # का ends in a matra (aa), not an inherent schwa, so nothing is dropped.
    assert romanize("का") == "kaa"


def test_independent_vowels():
    assert romanize("आम") == "aam"        # आ + म, final schwa dropped
    assert romanize("भारत") == "bhaarat"  # भ + ा = bhaa, र = ra, त = ta -> bhaarat (aa scheme)


def test_non_devanagari_passes_through():
    assert romanize("kal") == "kal"


def test_anusvara_is_homorganic():
    assert romanize("भूकंप") == "bhuukamp"  # anusvara before प -> m, not n
    assert romanize("हिंदी") == "hindii"    # anusvara before द -> n
