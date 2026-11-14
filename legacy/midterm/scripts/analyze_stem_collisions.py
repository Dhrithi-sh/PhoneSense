import json
from collections import defaultdict, Counter
from pathlib import Path

from text.normalize import normalize
from text.tokenize import tokenize
from text.stem import stem


INPUT_PATH = Path("data/dev_news.jsonl")
OUTPUT_PATH = Path("data/stem_collisions.tsv")

MIN_CLASS_FREQUENCY = 5
MIN_FORM_FREQUENCY = 2


def iter_articles(path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            yield json.loads(line)


def collect_document_forms(path):
    """
    Build:

        stem -> surface form -> set(doc_ids)

    This lets us measure whether different surface forms
    belonging to the same stem class occur in the same articles.
    """
    classes = defaultdict(lambda: defaultdict(set))

    for article in iter_articles(path):
        doc_id = article["doc_id"]

        text = (
            f"{article.get('headline', '')} "
            f"{article.get('body', '')}"
        )

        tokens = tokenize(normalize(text))

        # Avoid counting the same surface form multiple times
        # within one document.
        document_forms = set()

        for token in tokens:
            if len(token) < 2:
                continue

            if token.isdigit():
                continue

            stemmed = stem(token)

            if token != stemmed:
                document_forms.add((stemmed, token))

        for stemmed, surface in document_forms:
            classes[stemmed][surface].add(doc_id)

    return classes


def analyse_classes(classes):
    results = []

    for stemmed, forms in classes.items():
        form_frequencies = {
            surface: len(doc_ids)
            for surface, doc_ids in forms.items()
        }

        class_frequency = sum(form_frequencies.values())

        if class_frequency < MIN_CLASS_FREQUENCY:
            continue

        # Keep only surface forms that occur in at least
        # MIN_FORM_FREQUENCY documents.
        common_forms = {
            surface: doc_ids
            for surface, doc_ids in forms.items()
            if len(doc_ids) >= MIN_FORM_FREQUENCY
        }

        if len(common_forms) < 2:
            continue

        # Total number of documents containing at least one
        # member of this stem class.
        class_docs = set()

        for doc_ids in common_forms.values():
            class_docs.update(doc_ids)

        class_doc_count = len(class_docs)

        # Count pairwise same-document collisions.
        pairs = []

        surfaces = sorted(common_forms)

        for i in range(len(surfaces)):
            for j in range(i + 1, len(surfaces)):
                left = surfaces[i]
                right = surfaces[j]

                overlap = (
                    common_forms[left]
                    & common_forms[right]
                )

                left_docs = len(common_forms[left])
                right_docs = len(common_forms[right])

                smaller_frequency = min(
                    left_docs,
                    right_docs,
                )

                if smaller_frequency == 0:
                    collision_rate = 0.0
                else:
                    collision_rate = (
                        len(overlap)
                        / smaller_frequency
                    )

                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "overlap": len(overlap),
                        "collision_rate": collision_rate,
                    }
                )

        # Maximum collision among any pair.
        max_collision = max(
            pair["collision_rate"]
            for pair in pairs
        )

        total_overlap = sum(
            pair["overlap"]
            for pair in pairs
        )

        if max_collision == 0:
            classification = "SAFE"
        elif max_collision <= 0.10:
            classification = "LIKELY_SAFE"
        elif max_collision <= 0.30:
            classification = "REVIEW"
        else:
            classification = "UNSAFE"

        results.append(
            {
                "stem": stemmed,
                "forms": common_forms,
                "class_docs": class_doc_count,
                "max_collision": max_collision,
                "total_overlap": total_overlap,
                "classification": classification,
                "pairs": pairs,
            }
        )

    results.sort(
        key=lambda item: (
            -item["max_collision"],
            -item["class_docs"],
            item["stem"],
        )
    )

    return results


def print_report(results):
    print("=" * 90)
    print("SELECTIVE STEMMING COLLISION ANALYSIS")
    print("=" * 90)

    print(f"Candidate stem classes: {len(results)}")

    counts = Counter(
        result["classification"]
        for result in results
    )

    print()
    print("CLASSIFICATION SUMMARY")
    print("-" * 90)

    for label in (
        "SAFE",
        "LIKELY_SAFE",
        "REVIEW",
        "UNSAFE",
    ):
        print(
            f"{label:<15} "
            f"{counts[label]}"
        )

    print()
    print("HIGHEST-COLLISION CLASSES")
    print("-" * 90)

    for result in results[:100]:
        forms = result["forms"]

        form_text = ", ".join(
            f"{surface}({len(doc_ids)})"
            for surface, doc_ids
            in sorted(
                forms.items(),
                key=lambda item: -len(item[1]),
            )
        )

        print(
            f"{result['stem']:<20} "
            f"docs={result['class_docs']:<5} "
            f"collision={result['max_collision']:.3f} "
            f"{result['classification']:<12} "
            f"{form_text}"
        )

    print()
    print("SAFER MULTI-FORM CLASSES")
    print("-" * 90)

    safe_results = [
        result
        for result in results
        if result["classification"] in {
            "SAFE",
            "LIKELY_SAFE",
        }
    ]

    safe_results.sort(
        key=lambda item: -item["class_docs"]
    )

    for result in safe_results[:100]:
        forms = result["forms"]

        form_text = ", ".join(
            f"{surface}({len(doc_ids)})"
            for surface, doc_ids
            in sorted(
                forms.items(),
                key=lambda item: -len(item[1]),
            )
        )

        print(
            f"{result['stem']:<20} "
            f"docs={result['class_docs']:<5} "
            f"collision={result['max_collision']:.3f} "
            f"{result['classification']:<12} "
            f"{form_text}"
        )


def save_report(results, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "stem\tclassification\t"
            "class_docs\tmax_collision\t"
            "total_overlap\tforms\n"
        )

        for result in results:
            forms = ",".join(
                sorted(result["forms"])
            )

            file.write(
                f"{result['stem']}\t"
                f"{result['classification']}\t"
                f"{result['class_docs']}\t"
                f"{result['max_collision']:.6f}\t"
                f"{result['total_overlap']}\t"
                f"{forms}\n"
            )

    print()
    print(
        f"Full collision report written to: "
        f"{output_path}"
    )


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {INPUT_PATH}"
        )

    classes = collect_document_forms(INPUT_PATH)
    results = analyse_classes(classes)

    print_report(results)
    save_report(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()