import json

import pytest

from index.build import (
    build_index,
    iter_articles,
)


def sample_articles():
    return [
        {
            "doc_id": "doc1",
            "headline": "पहली खबर",
            "body": "यह पहली खबर है।",
            "source": "test",
            "date": "2026-01-01",
            "state": "Delhi",
            "city": "Delhi",
            "section": "news",
            "dup_of": None,
            "links": [],
        },
        {
            "doc_id": "doc2",
            "headline": "दूसरी खबर",
            "body": "यह दूसरी खबर है।",
            "source": "test",
            "date": "2026-01-02",
            "state": "Delhi",
            "city": "Delhi",
            "section": "news",
            "dup_of": None,
            "links": [],
        },
    ]


def write_jsonl(path, articles):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for article in articles:
            file.write(
                json.dumps(
                    article,
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_iter_articles_reads_valid_jsonl(tmp_path):
    input_path = tmp_path / "news.jsonl"

    articles = sample_articles()

    write_jsonl(
        input_path,
        articles,
    )

    result = list(
        iter_articles(input_path)
    )

    assert result == articles


def test_iter_articles_skips_blank_lines(tmp_path):
    input_path = tmp_path / "news.jsonl"

    articles = sample_articles()

    with input_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n")
        file.write(
            json.dumps(
                articles[0],
                ensure_ascii=False,
            )
            + "\n"
        )
        file.write("\n")
        file.write(
            json.dumps(
                articles[1],
                ensure_ascii=False,
            )
            + "\n"
        )
        file.write("\n")

    result = list(
        iter_articles(input_path)
    )

    assert result == articles


def test_iter_articles_missing_file(tmp_path):
    input_path = (
        tmp_path / "missing.jsonl"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Article file not found",
    ):
        list(
            iter_articles(input_path)
        )


def test_iter_articles_invalid_json(tmp_path):
    input_path = tmp_path / "news.jsonl"

    with input_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            '{"doc_id": "doc1",\n'
        )
        file.write(
            'this is invalid json\n'
        )

    with pytest.raises(
        ValueError,
        match="Line 1: invalid JSON",
    ):
        list(
            iter_articles(input_path)
        )


def test_iter_articles_requires_json_object(tmp_path):
    input_path = tmp_path / "news.jsonl"

    with input_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                ["not", "an", "object"]
            )
            + "\n"
        )

    with pytest.raises(
        ValueError,
        match="article must be a JSON object",
    ):
        list(
            iter_articles(input_path)
        )


@pytest.mark.parametrize(
    "field",
    [
        "doc_id",
        "headline",
        "body",
    ],
)
def test_iter_articles_requires_required_fields(
    tmp_path,
    field,
):
    input_path = tmp_path / "news.jsonl"

    article = sample_articles()[0]

    del article[field]

    write_jsonl(
        input_path,
        [article],
    )

    with pytest.raises(
        ValueError,
        match=f"missing required field '{field}'",
    ):
        list(
            iter_articles(input_path)
        )


@pytest.mark.parametrize(
    "field",
    [
        "doc_id",
        "headline",
        "body",
    ],
)
def test_iter_articles_requires_string_fields(
    tmp_path,
    field,
):
    input_path = tmp_path / "news.jsonl"

    article = sample_articles()[0]

    article[field] = 123

    write_jsonl(
        input_path,
        [article],
    )

    with pytest.raises(
        ValueError,
        match=f"'{field}' must be a string",
    ):
        list(
            iter_articles(input_path)
        )


def test_build_index_none_mode(tmp_path):
    input_path = tmp_path / "news.jsonl"

    write_jsonl(
        input_path,
        sample_articles(),
    )

    index = build_index(
        input_path=input_path,
        mode="none",
    )

    assert index.mode == "none"
    assert index.N == 2
    assert "पहली" in index.vocab
    assert "दूसरी" in index.vocab


def test_build_index_light_mode(tmp_path):
    input_path = tmp_path / "news.jsonl"

    write_jsonl(
        input_path,
        sample_articles(),
    )

    index = build_index(
        input_path=input_path,
        mode="light",
    )

    assert index.mode == "light"
    assert index.N == 2


def test_build_index_aggr_mode(tmp_path):
    input_path = tmp_path / "news.jsonl"

    write_jsonl(
        input_path,
        sample_articles(),
    )

    index = build_index(
        input_path=input_path,
        mode="aggr",
    )

    assert index.mode == "aggr"
    assert index.N == 2


def test_build_index_auto_mode(tmp_path):
    input_path = tmp_path / "news.jsonl"

    write_jsonl(
        input_path,
        sample_articles(),
    )

    index = build_index(
        input_path=input_path,
        mode="auto",
    )

    assert index.mode == "auto"
    assert index.N == 2


def test_build_index_invalid_mode(tmp_path):
    input_path = tmp_path / "news.jsonl"

    write_jsonl(
        input_path,
        sample_articles(),
    )

    with pytest.raises(
        ValueError,
        match="Unsupported index mode",
    ):
        build_index(
            input_path=input_path,
            mode="invalid",
        )


def test_build_index_preserves_metadata(tmp_path):
    input_path = tmp_path / "news.jsonl"

    articles = sample_articles()

    write_jsonl(
        input_path,
        articles,
    )

    index = build_index(
        input_path=input_path,
        mode="none",
    )

    assert index.meta["doc1"]["source"] == "test"
    assert index.meta["doc1"]["date"] == "2026-01-01"
    assert index.meta["doc1"]["state"] == "Delhi"
    assert index.meta["doc1"]["city"] == "Delhi"
    assert index.meta["doc1"]["section"] == "news"
    assert index.meta["doc1"]["dup_of"] is None
    assert index.meta["doc1"]["links"] == []


def test_build_index_writes_output_file(tmp_path):
    input_path = tmp_path / "news.jsonl"
    output_path = tmp_path / "index.pkl"

    write_jsonl(
        input_path,
        sample_articles(),
    )

    index = build_index(
        input_path=input_path,
        mode="none",
        output_path=output_path,
    )

    assert output_path.exists()
    assert index.N == 2


def test_build_index_skips_duplicate_documents(
    tmp_path,
    capsys,
):
    input_path = tmp_path / "news.jsonl"

    articles = sample_articles()

    duplicate = articles[0].copy()
    articles.append(duplicate)

    write_jsonl(
        input_path,
        articles,
    )

    index = build_index(
        input_path=input_path,
        mode="none",
    )

    captured = capsys.readouterr()

    assert (
        "Warning: duplicate doc_id 'doc1'"
        in captured.out
    )

    assert (
        "Skipping duplicate article"
        in captured.out
    )

    assert (
        "Skipped 1 duplicate document(s)."
        in captured.out
    )

    assert index.N == 2

    assert "doc1" in index.meta
    assert "doc2" in index.meta

    assert len(index.meta) == 2


def test_build_index_keeps_first_duplicate_document(
    tmp_path,
):
    input_path = tmp_path / "news.jsonl"

    articles = sample_articles()

    duplicate = articles[0].copy()

    duplicate["headline"] = (
        "THIS SHOULD NOT REPLACE "
        "THE FIRST ARTICLE"
    )

    duplicate["body"] = (
        "THIS SHOULD NOT REPLACE "
        "THE FIRST ARTICLE"
    )

    articles.append(duplicate)

    write_jsonl(
        input_path,
        articles,
    )

    index = build_index(
        input_path=input_path,
        mode="none",
    )

    assert index.N == 2

    assert "doc1" in index.meta
    assert "doc2" in index.meta

    assert (
        index.text["doc1"]["headline"]
        == "पहली खबर"
    )

    assert (
        index.text["doc1"]["body"]
        == "यह पहली खबर है।"
    )