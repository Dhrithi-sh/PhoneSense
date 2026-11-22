### Test split

| System | Parent | Δ nDCG@10 | p | W / L / T |
|---|---|---:|---:|---:|
| BM25 (k1=1.2, b=0.75) | tfidf_lnc.ltc | +0.109 | 0.0665 | 9 / 7 / 0 |
| BM25 tuned | bm25_default | +0.010 | 0.8815 | 4 / 8 / 4 |
| BM25F tuned | bm25_tuned | -0.194 | 0.0004 | 1 / 14 / 1 |
| + filters | bm25_tuned | +0.030 | 0.1213 | 4 / 0 / 12 |
| + phrase | bm25_tuned | +0.000 | 1.0000 | 0 / 0 / 16 |
| + filters + phrase | bm25_tuned+phrase | +0.030 | 0.1213 | 4 / 0 / 12 |
| Dense | bm25_tuned | pending (B) | - | - |
| Rocchio PRF | bm25_tuned | pending (B) | - | - |
| HyDE | bm25_tuned | pending (B) | - | - |
| RRF (BM25, dense) | bm25_tuned | pending (B) | - | - |
| Weighted fusion | bm25_tuned | pending (B) | - | - |
| PageRank rerank | bm25_tuned | -0.003 | 0.7838 | 3 / 3 / 10 |
| Personalized PR | bm25_tuned | +0.008 | 0.1452 | 6 / 1 / 9 |
| Trust-weighted rerank | bm25_tuned | +0.011 | 0.0384 | 8 / 1 / 7 |

p: paired randomization test (two-sided, 10,000 permutations, uncorrected). W/L/T: queries where the system is better / worse / tied.

### Dev split

| System | Parent | Δ nDCG@10 | p | W / L / T |
|---|---|---:|---:|---:|
| BM25 (k1=1.2, b=0.75) | tfidf_lnc.ltc | +0.065 | 0.2741 | 14 / 8 / 1 |
| BM25 tuned | bm25_default | +0.013 | 0.2764 | 8 / 10 / 5 |
| BM25F tuned | bm25_tuned | -0.270 | 0.0001 | 3 / 20 / 0 |
| + filters | bm25_tuned | -0.038 | 0.2134 | 3 / 6 / 14 |
| + phrase | bm25_tuned | -0.002 | 0.4999 | 0 / 2 / 21 |
| + filters + phrase | bm25_tuned+phrase | -0.038 | 0.2134 | 3 / 6 / 14 |
| Dense | bm25_tuned | pending (B) | - | - |
| Rocchio PRF | bm25_tuned | pending (B) | - | - |
| HyDE | bm25_tuned | pending (B) | - | - |
| RRF (BM25, dense) | bm25_tuned | pending (B) | - | - |
| Weighted fusion | bm25_tuned | pending (B) | - | - |
| PageRank rerank | bm25_tuned | -0.019 | 0.0409 | 2 / 7 / 14 |
| Personalized PR | bm25_tuned | -0.014 | 0.1207 | 6 / 5 / 12 |
| Trust-weighted rerank | bm25_tuned | -0.011 | 0.2606 | 4 / 9 / 10 |

p: paired randomization test (two-sided, 10,000 permutations, uncorrected). W/L/T: queries where the system is better / worse / tied.
