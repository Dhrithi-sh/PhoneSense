"""Build data/processed/{phones,reviews}.jsonl from the raw dataset.

Placeholder until T02 (owner: A). Paths come from config.yaml via phonesense.config.
"""

from phonesense.config import get_path


def main() -> None:
    print("TODO (T02): build phone/review files from the raw dataset.")
    print(f"  will write: {get_path('phones')}")
    print(f"  will write: {get_path('reviews')}")


if __name__ == "__main__":
    main()
