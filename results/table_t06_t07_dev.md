| Run | P@10 | MAP | nDCG@10 | judged@10 |
|---|---:|---:|---:|---:|
| bm25_tuned_dev (baseline) | **0.265** | 0.441 | **0.600** | **0.696** |
| bm25_tuned+filters_dev | 0.248 | 0.382 | 0.562 | 0.543 |
| bm25_tuned+phrase_dev | **0.265** | **0.470** | 0.598 | 0.683 |
| bm25_tuned+filters+phrase_dev | 0.248 | 0.411 | 0.560 | 0.530 |

† p < 0.05 vs. bm25_tuned_dev (paired randomization test, 10000 permutations, Holm-corrected). n = 23 queries.
