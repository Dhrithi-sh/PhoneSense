import json
from collections import defaultdict, Counter
from pathlib import Path

from text.normalize import normalize
from text.tokenize import tokenize
from text.stem import stem


INPUT_PATH = Path("data/dev_news.jsonl")

# Only show classes where the corpus has enough evidence.
MIN_STEM_FREQUENCY = 5

# Maximum number of surface forms to print for one stem.
MAX_FORMS_PER_STEM = 20


def iter_articles(path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            yield json.loads(line)


def collect_stem_classes(path):
    """
    Build:

        stem -> surface form -> frequency

    using the actual corpus and the existing light stemmer.
    """
    classes = defaultdict(Counter)

    total_tokens = 0

    for article in iter_articles(path):
        text = f"{article.get('headline', '')} {article.get('body', '')}"

        normalized = normalize(text)
        tokens = tokenize(normalized)

        for token in tokens:
            total_tokens += 1

            stemmed = stem(token)

            # Ignore numbers and very short tokens for this analysis.
            if len(token) < 2:
                continue

            if token.isdigit():
                continue

            classes[stemmed][token] += 1

    return classes, total_tokens


def print_summary(classes, total_tokens):
    multi_form_classes = []

    for stemmed, forms in classes.items():
        if sum(forms.values()) < MIN_STEM_FREQUENCY:
            continue

        if len(forms) >= 2:
            multi_form_classes.append((stemmed, forms))

    multi_form_classes.sort(
        key=lambda item: sum(item[1].values()),
        reverse=True,
    )

    print("=" * 80)
    print("SELECTIVE STEMMING ANALYSIS")
    print("=" * 80)

    print(f"Total analyzed tokens: {total_tokens}")
    print(f"Total stem classes: {len(classes)}")
    print(
        f"Multi-form classes with frequency >= {MIN_STEM_FREQUENCY}: "
        f"{len(multi_form_classes)}"
    )

    print()
    print("TOP STEM CLASSES")
    print("-" * 80)

    for stemmed, forms in multi_form_classes[:100]:
        total = sum(forms.values())

        sorted_forms = sorted(
            forms.items(),
            key=lambda item: (-item[1], item[0]),
        )

        display_forms = sorted_forms[:MAX_FORMS_PER_STEM]

        form_text = ", ".join(
            f"{surface}({frequency})"
            for surface, frequency in display_forms
        )

        if len(sorted_forms) > MAX_FORMS_PER_STEM:
            form_text += ", ..."

        print(
            f"{stemmed:<20} "
            f"total={total:<6} "
            f"forms={len(forms):<3} "
            f"{form_text}"
        )


def save_report(classes, output_path):
    """
    Save the complete stem-class analysis to a TSV file so it can
    be inspected or used for later selective-auto decisions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for stemmed, forms in classes.items():
        total = sum(forms.values())

        if total < MIN_STEM_FREQUENCY:
            continue

        for surface, frequency in forms.items():
            rows.append(
                (
                    stemmed,
                    surface,
                    frequency,
                    len(forms),
                    total,
                )
            )

    rows.sort(
        key=lambda row: (-row[4], row[0], -row[2], row[1])
    )

    with output_path.open("w", encoding="utf-8") as file:
        file.write(
            "stem\tsurface_form\tfrequency\tform_count\tclass_frequency\n"
        )

        for stemmed, surface, frequency, form_count, total in rows:
            file.write(
                f"{stemmed}\t"
                f"{surface}\t"
                f"{frequency}\t"
                f"{form_count}\t"
                f"{total}\n"
            )

    print()
    print(f"Full report written to: {output_path}")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {INPUT_PATH}"
        )

    classes, total_tokens = collect_stem_classes(INPUT_PATH)

    print_summary(classes, total_tokens)

    save_report(
        classes,
        Path("data/stem_classes.tsv"),
    )


if __name__ == "__main__":
    main()