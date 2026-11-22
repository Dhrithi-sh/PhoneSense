| Run | P@10 | R@10 | MAP | nDCG@10 | RR | judged@10 |
|---|---:|---:|---:|---:|---:|---:|
| tfidf_lnc.ltc_dev (baseline) | **0.317** | 0.521 | 0.374 | 0.522 | 0.596 | **0.796** |
| bm25_default_dev | 0.265 | 0.574 | 0.427 | 0.588 | 0.714 | 0.648 |
| bm25_tuned_dev | 0.265 | **0.586** | **0.441** | **0.600** | **0.743** | 0.696 |
| bm25f_tuned_dev | 0.143† | 0.269† | 0.235 | 0.330† | 0.495 | 0.326 |

† p < 0.05 vs. tfidf_lnc.ltc_dev (paired randomization test, 10000 permutations, Holm-corrected). n = 23 queries.
