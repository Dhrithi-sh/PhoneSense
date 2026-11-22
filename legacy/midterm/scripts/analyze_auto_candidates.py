import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from text.normalize import normalize
from text.tokenize import tokenize
from text.stem import stem


INPUT_PATH = Path("data/news_dedup/news_dedup.jsonl")
OUTPUT_PATH = Path("artifacts/auto_candidates.tsv")

MIN_CLASS_FREQUENCY = 5
MIN_FORM_FREQUENCY = 2


def levenshtein_distance(a, b):
    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)

            current.append(
                min(insert_cost, delete_cost, replace_cost)
            )

        previous = current

    return previous[-1]


def similarity(a, b):
    if not a and not b:
        return 1.0

    distance = levenshtein_distance(a, b)
    return 1.0 - (distance / max(len(a), len(b)))


def load_corpus():
    stem_forms = defaultdict(Counter)
    stem_docs = defaultdict(lambda: defaultdict(set))
    total_tokens = 0

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            import json

            article = json.loads(line)

            doc_id = article["doc_id"]

            text = (
                article.get("headline", "")
                + " "
                + article.get("body", "")
            )

            tokens = tokenize(normalize(text))

            for token in tokens:
                total_tokens += 1

                # Ignore very short tokens and pure numbers.
                if len(token) < 3:
                    continue

                if token.isdigit():
                    continue

                stemmed = stem(token)

                if token == stemmed:
                    continue

                stem_forms[stemmed][token] += 1
                stem_docs[stemmed][token].add(doc_id)

    return stem_forms, stem_docs, total_tokens


def classify_candidate(
    stem_word,
    forms,
    stem_docs,
):
    total_frequency = sum(forms.values())

    if total_frequency < MIN_CLASS_FREQUENCY:
        return None

    if len(forms) < 2:
        return None

    if any(
        frequency < MIN_FORM_FREQUENCY
        for frequency in forms.values()
    ):
        return None

    form_items = list(forms.items())

    similarities = []

    for i in range(len(form_items)):
        form_a = form_items[i][0]

        for j in range(i + 1, len(form_items)):
            form_b = form_items[j][0]

            similarities.append(
                similarity(form_a, form_b)
            )

    avg_similarity = (
        sum(similarities) / len(similarities)
        if similarities
        else 0.0
    )

    min_similarity = (
        min(similarities)
        if similarities
        else 0.0
    )

    max_collision = 0.0
    collision_pairs = []

    for i in range(len(form_items)):
        form_a, _ = form_items[i]

        for j in range(i + 1, len(form_items)):
            form_b, _ = form_items[j]

            docs_a = stem_docs[form_a]
            docs_b = stem_docs[form_b]

            if not docs_a or not docs_b:
                continue

            overlap = len(docs_a & docs_b)
            union = len(docs_a | docs_b)

            collision = (
                overlap / union
                if union
                else 0.0
            )

            max_collision = max(
                max_collision,
                collision,
            )

            collision_pairs.append(
                (
                    collision,
                    form_a,
                    form_b,
                )
            )

    # High similarity + low collision = good candidate.
    if (
        avg_similarity >= 0.65
        and min_similarity >= 0.50
        and max_collision <= 0.10
    ):
        classification = "STRONG"

    elif (
        avg_similarity >= 0.55
        and min_similarity >= 0.40
        and max_collision <= 0.20
    ):
        classification = "GOOD"

    elif (
        avg_similarity >= 0.45
        and max_collision <= 0.30
    ):
        classification = "REVIEW"

    else:
        classification = "REJECT"

    worst_pair = max(
        collision_pairs,
        default=(0.0, "", ""),
    )

    return {
        "stem": stem_word,
        "total_frequency": total_frequency,
        "forms": len(forms),
        "avg_similarity": avg_similarity,
        "min_similarity": min_similarity,
        "max_collision": max_collision,
        "worst_form_a": worst_pair[1],
        "worst_form_b": worst_pair[2],
        "classification": classification,
        "surface_forms": ", ".join(
            f"{form}:{frequency}"
            for form, frequency
            in forms.most_common(10)
        ),
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {INPUT_PATH}"
        )

    stem_forms, stem_docs, total_tokens = load_corpus()

    candidates = []

    for stem_word, forms in stem_forms.items():
        result = classify_candidate(
            stem_word,
            forms,
            stem_docs[stem_word],
        )

        if result is not None:
            candidates.append(result)

    candidates.sort(
        key=lambda row: (
            {
                "STRONG": 0,
                "GOOD": 1,
                "REVIEW": 2,
                "REJECT": 3,
            }[row["classification"]],
            -row["total_frequency"],
        )
    )

    counts = Counter(
        row["classification"]
        for row in candidates
    )

    print(f"Total analyzed tokens: {total_tokens:,}")
    print(f"Candidate stem classes: {len(candidates):,}")
    print()
    print("CLASSIFICATION")
    print("-" * 60)

    for label in (
        "STRONG",
        "GOOD",
        "REVIEW",
        "REJECT",
    ):
        print(
            f"{label:<10} "
            f"{counts[label]:>6}"
        )

    print()
    print("TOP STRONG CANDIDATES")
    print("-" * 80)

    strong = [
        row
        for row in candidates
        if row["classification"] == "STRONG"
    ]

    for row in strong[:30]:
        print(
            f"{row['stem']:<15} "
            f"freq={row['total_frequency']:<6} "
            f"forms={row['forms']:<3} "
            f"sim={row['avg_similarity']:.2f} "
            f"collision={row['max_collision']:.2f} "
            f"{row['surface_forms']}"
        )

    print()
    print("TOP GOOD CANDIDATES")
    print("-" * 80)

    good = [
        row
        for row in candidates
        if row["classification"] == "GOOD"
    ]

    for row in good[:30]:
        print(
            f"{row['stem']:<15} "
            f"freq={row['total_frequency']:<6} "
            f"forms={row['forms']:<3} "
            f"sim={row['avg_similarity']:.2f} "
            f"collision={row['max_collision']:.2f} "
            f"{row['surface_forms']}"
        )

    print()
    print("TOP REVIEW CANDIDATES")
    print("-" * 80)

    review = [
        row
        for row in candidates
        if row["classification"] == "REVIEW"
    ]

    for row in review[:30]:
        print(
            f"{row['stem']:<15} "
            f"freq={row['total_frequency']:<6} "
            f"forms={row['forms']:<3} "
            f"sim={row['avg_similarity']:.2f} "
            f"collision={row['max_collision']:.2f} "
            f"{row['surface_forms']}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "stem",
        "total_frequency",
        "forms",
        "avg_similarity",
        "min_similarity",
        "max_collision",
        "worst_form_a",
        "worst_form_b",
        "classification",
        "surface_forms",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter="\t",
        )

        writer.writeheader()

        for row in candidates:
            writer.writerow(row)

    print()
    print(f"Saved detailed results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()