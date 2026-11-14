from index.positional import Index
from index.query import search_query


QUERIES = [
    "बारिश",
    "भारी बारिश",
    "उत्तर प्रदेश",
    "जम्मू कश्मीर",
    "मौसम",
    "तेज बारिश",
    "बारिश की संभावना",
    "लोगों की मौत",
]


def main():
    indexes = {
        "none": Index.load("none"),
        "light": Index.load("light"),
        "aggr": Index.load("aggr"),
    }

    for query in QUERIES:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        for mode, idx in indexes.items():
            results = search_query(
                idx,
                query,
                mode=mode,
                zone="body",
            )

            print(f"\n{mode.upper()}: {len(results)} results")

            for doc_id in results[:5]:
                print(f"  {doc_id}")


if __name__ == "__main__":
    main()