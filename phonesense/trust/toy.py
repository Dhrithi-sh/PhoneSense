"""Seeded toy reviews in the contract format, for tests and for dry runs before real data.

Each phone has a hidden true quality; each user rates 1 to 8 phones as that quality plus
noise (10% of users are careless, with more noise). Nothing here is real data.
"""

from __future__ import annotations

import numpy as np

from phonesense.schemas import Review, make_review_id

BASE_TIMESTAMP_MS = 1_600_000_000_000


def toy_doc_id(i: int) -> str:
    return f"B0TOY{i:05d}"


def make_toy_reviews(seed: int = 7, n_phones: int = 30, n_users: int = 200) -> list[Review]:
    rng = np.random.default_rng(seed)
    quality = np.clip(rng.normal(3.8, 0.6, n_phones), 1, 5)
    reviews: list[Review] = []
    for u in range(n_users):
        user_id = f"TOYU{u:05d}"
        noise = 1.5 if rng.random() < 0.1 else 0.6
        for p in rng.choice(n_phones, size=int(rng.integers(1, 9)), replace=False):
            doc_id = toy_doc_id(int(p))
            ts = BASE_TIMESTAMP_MS + 1000 * len(reviews)
            reviews.append(Review(
                review_id=make_review_id(user_id, doc_id, ts), doc_id=doc_id, user_id=user_id,
                rating=int(np.clip(np.rint(quality[p] + rng.normal(0, noise)), 1, 5)),
                title="", text="", timestamp=ts,
                helpful_votes=int(rng.integers(0, 4)), verified=True,
            ))
    return reviews
