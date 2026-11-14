from dhvani.query.phonetics import dhvani_code, soundex


# --- classic Soundex (lecture / Manning IIR example values) ---

def test_soundex_known_values():
    assert soundex("Hermann") == "H655"
    assert soundex("Robert") == "R163"
    assert soundex("Rupert") == "R163"   # sounds like Robert -> same code
    assert soundex("Rubin") == "R150"


def test_soundex_is_padded_to_four():
    assert soundex("Lee") == "L000"


def test_soundex_collapses_mausam_and_mosam():
    # Even plain Soundex buckets these together.
    assert soundex("mausam") == soundex("mosam")


def test_soundex_empty_for_no_letters():
    assert soundex("123") == ""


# --- Dhvani-code (our Hindi Soundex) ---

def test_dhvani_code_spec_example():
    # The number quoted in the plan: म=5, स=8, म=5.
    assert dhvani_code("मौसम") == "585"
    assert dhvani_code("mausam") == "585"
    assert dhvani_code("mosam") == "585"


def test_dhvani_code_is_script_agnostic():
    # Same consonant skeleton across scripts: ब=9, र=6, श=8.
    assert dhvani_code("बारिश") == dhvani_code("baarish") == "968"


def test_dhvani_code_collapses_doubled_consonants():
    # पत्ता / patta: the doubled t is one class digit.
    assert dhvani_code("patta") == dhvani_code("पत्ता")


def test_dhvani_code_drops_vowels_only():
    # A pure-vowel string has no consonant skeleton.
    assert dhvani_code("आओ") == ""


def test_anusvara_is_homorganic_nasal():
    # anusvara before a labial is म (so भूकंप codes like भूकम्प and "bhukamp"),
    # elsewhere it is न (हिंदी like हिन्दी).
    assert dhvani_code("भूकंप") == dhvani_code("भूकम्प") == dhvani_code("bhukamp")
    assert dhvani_code("हिंदी") == dhvani_code("हिन्दी")
