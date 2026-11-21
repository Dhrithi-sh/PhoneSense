import math

from dhvani.rank.sample_index import SampleIndex, tokenize


def test_tokenizer_keeps_vowel_signs_together():
    assert tokenize("कल का मौसम") == ["कल", "का", "मौसम"]


def test_basic_counts():
    idx = SampleIndex.load("none")
    assert idx.N == 20
    assert "मौसम" in idx.vocab
    # मौसम appears in 4 articles: jagran_1001, nbt_2001, amarujala_3001, aajtak_5001
    assert idx.df("मौसम") == 4
    assert idx.df("नहींमिलेगा") == 0


def test_postings_have_positions_and_are_sorted():
    idx = SampleIndex.load("none")
    body = idx.postings("मौसम", "body")
    doc_ids = [doc_id for doc_id, _, _ in body]
    assert doc_ids == sorted(doc_ids)
    nbt = dict((d, (tf, pos)) for d, tf, pos in body)["nbt_2001"]
    assert nbt[0] == 2 and len(nbt[1]) == 2


def test_zones_are_separate():
    idx = SampleIndex.load("none")
    headline_docs = {d for d, _, _ in idx.postings("मौसम", "headline")}
    assert headline_docs == {"nbt_2001", "aajtak_5001"}


def test_doc_norm_is_lnc_length():
    idx = SampleIndex.load("none")
    for doc_id, norm in idx.doc_norm.items():
        assert norm > 0 and math.isfinite(norm)


def test_meta_has_fields_for_filters_and_pagerank():
    idx = SampleIndex.load("none")
    m = idx.meta["amarujala_3004"]
    assert m["dup_of"] == "aajtak_5004"
    assert m["source"] == "amarujala"
    assert m["links"] == ["aajtak_5004"]
