from index.positional import Index


def test_add_document_and_postings():
    idx = Index("none")

    idx.add_document(
        doc_id="doc1",
        headline="बारिश आज",
        body="आज दिल्ली में बारिश हुई",
        metadata={
            "source": "jagran",
            "date": "2026-10-06",
            "state": "delhi",
            "section": "weather",
            "dup_of": None,
        },
    )

    assert idx.N == 1
    assert "बारिश" in idx.vocab

    assert idx.postings(
        "बारिश",
        "headline",
    ) == [
        ("doc1", 1, [0])
    ]

    assert idx.postings(
        "बारिश",
        "body",
    ) == [
        ("doc1", 1, [3])
    ]


def test_postings_for_alias():
    idx = Index("none")

    idx.add_document(
        "doc1",
        "बारिश आज",
        "आज बारिश हुई",
        {},
    )

    assert idx.postings_for(
        "बारिश",
        "headline",
    ) == idx.postings(
        "बारिश",
        "headline",
    )


def test_document_frequency_counts_documents_not_occurrences():
    idx = Index("none")

    idx.add_document(
        "doc1",
        "बारिश आज",
        "आज बारिश हुई बारिश",
        {},
    )

    idx.add_document(
        "doc2",
        "बारिश",
        "बारिश जारी है",
        {},
    )

    assert idx.df("बारिश") == 2


def test_light_index_uses_stems():
    idx = Index("light")

    idx.add_document(
        "doc1",
        "लड़कियों की खबर",
        "किताबों की बात",
        {},
    )

    assert "लड़क" in idx.vocab
    assert "किताब" in idx.vocab


def test_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Index,
        "INDEX_DIR",
        tmp_path,
    )

    idx = Index("none")

    idx.add_document(
        "doc1",
        "बारिश आज",
        "आज दिल्ली में बारिश हुई",
        {
            "source": "jagran",
            "date": "2026-10-06",
            "state": "delhi",
            "section": "weather",
            "dup_of": None,
        },
    )

    explicit_path = (
        tmp_path / "explicit.pkl"
    )

    idx.save(explicit_path)

    with open(
        explicit_path,
        "rb",
    ) as f:
        import pickle

        loaded = pickle.load(f)

    assert loaded.mode == "none"
    assert loaded.N == 1
    assert loaded.vocab == idx.vocab
    assert loaded.meta == idx.meta

    idx.save()

    loaded = Index.load("none")

    assert loaded.mode == "none"
    assert loaded.N == 1
    assert loaded.vocab == idx.vocab

    assert loaded.postings(
        "बारिश",
        "headline",
    ) == [
        ("doc1", 1, [0])
    ]