from index.positional import Index


THRESHOLDS = [0.90, 0.80, 0.70, 0.60, 0.50]


def get_df_counts(index: Index):
    """
    Return (term, document_frequency) pairs for every term
    in the index vocabulary.
    """
    results = []

    for term in index.vocab:
        results.append((term, index.df(term)))

    return results


def main():
    index = Index.load("light")

    df_counts = get_df_counts(index)

    vocabulary_size = len(df_counts)
    total_documents = index.N

    print("=" * 70)
    print("STOP-WORD DOCUMENT-FREQUENCY STATISTICS")
    print("=" * 70)

    print(f"Documents:  {total_documents}")
    print(f"Vocabulary: {vocabulary_size}")
    print()

    print(f"{'Threshold':<15}{'Terms':>10}{'Vocabulary %':>18}")
    print("-" * 45)

    for threshold in THRESHOLDS:
        minimum_df = threshold * total_documents

        matching_terms = [
            term
            for term, df in df_counts
            if df >= minimum_df
        ]

        count = len(matching_terms)
        percentage = (count / vocabulary_size) * 100

        print(
            f"DF >= {threshold:.0%}"
            f"{count:>14}"
            f"{percentage:>17.2f}%"
        )

    print()
    print("=" * 70)
    print("EXAMPLE TERMS AT EACH THRESHOLD")
    print("=" * 70)

    # Sort by DF from highest to lowest.
    df_counts.sort(key=lambda item: (-item[1], item[0]))

    for threshold in THRESHOLDS:
        minimum_df = threshold * total_documents

        matching_terms = [
            (term, df)
            for term, df in df_counts
            if df >= minimum_df
        ]

        print()
        print(f"DF >= {threshold:.0%}: {len(matching_terms)} terms")

        # Show at most the first 20.
        for term, df in matching_terms[:20]:
            percentage = (df / total_documents) * 100
            print(f"  {term:<20} {df:>4} ({percentage:>5.1f}%)")


if __name__ == "__main__":
    main()