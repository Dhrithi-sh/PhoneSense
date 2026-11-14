from collections import Counter

from index.positional import Index


def analyze_index(index: Index, top_n: int = 100):
    """
    Analyze the vocabulary of an existing index using document frequency.

    A term's document frequency (DF) is the number of distinct documents
    in which the term appears, across both headline and body.
    """

    df_counts = []

    for term in index.vocab:
        df = index.df(term)
        df_counts.append((term, df))

    df_counts.sort(key=lambda item: (-item[1], item[0]))

    return df_counts[:top_n]


def print_report(index: Index, top_n: int = 100):
    print("=" * 80)
    print("STOP-WORD / DOCUMENT-FREQUENCY ANALYSIS")
    print("=" * 80)

    print(f"Mode:        {index.mode}")
    print(f"Documents:   {index.N}")
    print(f"Vocabulary:  {len(index.vocab)}")
    print()

    results = analyze_index(index, top_n)

    print(f"{'Rank':<6}{'Term':<25}{'DF':>8}{'DF %':>10}")
    print("-" * 55)

    for rank, (term, df) in enumerate(results, start=1):
        percentage = (df / index.N) * 100

        print(
            f"{rank:<6}"
            f"{term:<25}"
            f"{df:>8}"
            f"{percentage:>9.2f}%"
        )


def main():
    # We use the light index because this is the stemming mode
    # we currently consider the reliable production baseline.
    index = Index.load("light")

    print_report(index, top_n=100)


if __name__ == "__main__":
    main()