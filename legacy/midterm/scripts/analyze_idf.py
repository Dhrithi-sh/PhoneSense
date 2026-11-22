import math

from index.positional import Index
from index.scoring import idf


TERMS = [
    "के",
    "में",
    "है",
    "और",
    "बारिश",
    "मौसम",
    "सरकार",
    "दिल्ली",
    "भारत",
]


def main():
    index = Index.load("light")

    print("=" * 70)
    print("REAL CORPUS IDF ANALYSIS")
    print("=" * 70)

    print(f"Documents: {index.N}")
    print()

    print(f"{'Term':<20}{'DF':>8}{'IDF':>12}")
    print("-" * 42)

    for term in TERMS:
        df = index.df(term)
        value = idf(index, term)

        print(
            f"{term:<20}"
            f"{df:>8}"
            f"{value:>12.4f}"
        )

    print()
    print("=" * 70)
    print("LOWEST-IDF TERMS")
    print("=" * 70)

    values = []

    for term in index.vocab:
        values.append(
            (term, index.df(term), idf(index, term))
        )

    values.sort(key=lambda item: (item[2], item[0]))

    for term, df, value in values[:20]:
        print(
            f"{term:<20}"
            f"DF={df:>4} "
            f"IDF={value:.4f}"
        )

    print()
    print("=" * 70)
    print("HIGHEST-IDF TERMS")
    print("=" * 70)

    values.sort(key=lambda item: (-item[2], item[0]))

    for term, df, value in values[:20]:
        print(
            f"{term:<20}"
            f"DF={df:>4} "
            f"IDF={value:.4f}"
        )


if __name__ == "__main__":
    main()