from index.positional import Index
from index.search import phrase_search, proximity_search


def main():
    idx = Index.load("none")

    print("=" * 70)
    print("REAL CORPUS PHRASE / POSITIONAL SEARCH TEST")
    print("=" * 70)
    print(f"Documents indexed: {idx.N}")
    print()

    # ---------------------------------------------------------
    # 1. Exact phrase search
    # ---------------------------------------------------------
    phrases = [
        ("भारी बारिश", ["भारी", "बारिश"]),
        ("उत्तर प्रदेश", ["उत्तर", "प्रदेश"]),
        ("जम्मू कश्मीर", ["जम्मू", "कश्मीर"]),
        ("तेज बारिश", ["तेज", "बारिश"]),
    ]

    for phrase, terms in phrases:
        results = phrase_search(
            idx,
            terms,
            zone="body",
        )

        print("-" * 70)
        print(f'PHRASE: "{phrase}"')
        print(f"Results: {len(results)}")

        for doc_id in results[:10]:
            print(f"  {doc_id}")

        print()

    # ---------------------------------------------------------
    # 2. Proximity search
    # ---------------------------------------------------------
    proximity_queries = [
        ("बारिश + मौसम", ["बारिश", "मौसम"], 3),
        ("बारिश + संभावना", ["बारिश", "संभावना"], 5),
        ("उत्तर + प्रदेश", ["उत्तर", "प्रदेश"], 5),
    ]

    for label, terms, distance in proximity_queries:
        results = proximity_search(
            idx,
            terms,
            distance=distance,
            zone="body",
        )

        print("-" * 70)
        print(f"PROXIMITY: {label}")
        print(f"Maximum distance: {distance}")
        print(f"Results: {len(results)}")

        for doc_id in results[:10]:
            print(f"  {doc_id}")

        print()

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()