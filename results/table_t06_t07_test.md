| Run | P@10 | MAP | nDCG@10 | judged@10 |
|---|---:|---:|---:|---:|
| bm25_tuned_test (baseline) | 0.381 | 0.591 | 0.663 | **0.619** |
| bm25_tuned+filters_test | **0.400** | **0.624** | **0.693** | 0.575 |
| bm25_tuned+phrase_test | 0.381 | 0.591 | 0.663 | **0.619** |
| bm25_tuned+filters+phrase_test | **0.400** | **0.624** | **0.693** | 0.575 |

† p < 0.05 vs. bm25_tuned_test (paired randomization test, 10000 permutations, Holm-corrected). n = 16 queries.
