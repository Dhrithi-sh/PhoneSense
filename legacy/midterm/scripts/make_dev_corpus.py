import hashlib
import json
from pathlib import Path

from datasets import load_dataset


OUTPUT = Path("data/dev_news.jsonl")
TARGET_ARTICLES = 1000


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def make_doc_id(article_id, index):
    article_id = clean_text(article_id)

    if article_id:
        return f"ilsum_{article_id}"

    return f"ilsum_{index:05d}"


def content_hash(headline, body):
    text = f"{headline}\n{body}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def main():
    print("Loading ILSUM Hindi dataset...")

    dataset = load_dataset(
        "ILSUM/ILSUM-2.0",
        "Hindi",
    )

    # Prefer the training split for our development corpus.
    split = dataset["train"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with OUTPUT.open("w", encoding="utf-8") as output:
        for index, article in enumerate(split):
            if written >= TARGET_ARTICLES:
                break

            headline = clean_text(
                article.get("Heading")
                or article.get("Headline")
            )

            body = clean_text(article.get("Article"))

            # Skip unusable records.
            if not headline or not body:
                continue

            doc_id = make_doc_id(article.get("id"), index)

            record = {
                "doc_id": doc_id,
                "url": None,
                "source": "ilsum",
                "section": None,
                "state": None,
                "city": None,
                "date": None,
                "headline": headline,
                "body": body,
                "keywords": [],
                "agency_flag": False,
                "content_hash": content_hash(headline, body),
                "dup_of": None,
                "links": [],
            }

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            written += 1

    print(f"Created: {OUTPUT}")
    print(f"Articles: {written}")


if __name__ == "__main__":
    main()