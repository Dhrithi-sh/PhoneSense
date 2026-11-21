from index.positional import Index


DF_THRESHOLD = 0.90


def main():
    index = Index.load("light")

    minimum_df = DF_THRESHOLD * index.N

    candidates = []

    for term in index.vocab:
        df = index.df(term)

        if df >= minimum_df:
            candidates.append((term, df))

    candidates.sort(key=lambda item: (-item[1], item[0]))

    print("=" * 70)
    print("STOP-WORD CANDIDATES")
    print("=" * 70)

    print(f"Documents: {index.N}")
    print(f"DF threshold: {DF_THRESHOLD:.0%}")
    print(f"Minimum DF: {minimum_df:.0f}")
    print(f"Candidates: {len(candidates)}")
    print()

    for rank, (term, df) in enumerate(candidates, start=1):
        percentage = (df / index.N) * 100
        print(f"{rank:>2}. {term:<20} DF={df:>4} ({percentage:.1f}%)")

    output_path = "data/stopword_candidates.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        for term, _ in candidates:
            file.write(term + "\n")

    print()
    print(f"Saved candidates to: {output_path}")


if __name__ == "__main__":
    main()