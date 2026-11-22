| Run | P@10 | R@10 | MAP | nDCG@10 | RR | judged@10 |
|---|---:|---:|---:|---:|---:|---:|
| tfidf_lnc.ltc_test (baseline) | 0.375 | 0.590 | 0.485 | 0.544 | 0.673 | **0.781** |
| bm25_default_test | 0.369 | 0.609 | 0.579 | 0.653 | **0.758** | 0.606 |
| bm25_tuned_test | **0.381** | **0.634** | **0.591** | **0.663** | 0.755 | 0.619 |
| bm25f_tuned_test | 0.263 | 0.470 | 0.424 | 0.469 | 0.701 | 0.369 |

† p < 0.05 vs. tfidf_lnc.ltc_test (paired randomization test, 10000 permutations, Holm-corrected). n = 16 queries.
