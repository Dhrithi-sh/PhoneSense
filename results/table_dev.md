| Run | P@10 | R@10 | MAP | nDCG@10 | RR | judged@10 |
|---|---:|---:|---:|---:|---:|---:|
| bm25_default_dev | 0.265 | 0.574 | 0.427 | 0.588 | 0.714 | 0.648 |
| bm25_tuned+filters+phrase_dev | 0.248 | 0.548 | 0.411 | 0.560 | 0.735 | 0.530 |
| bm25_tuned+filters_dev | 0.248 | 0.548 | 0.382 | 0.562 | 0.706 | 0.543 |
| bm25_tuned+pagerank0.1_dev | 0.248 | 0.566 | 0.434 | 0.582 | 0.739 | 0.648 |
| bm25_tuned+phrase_dev | 0.265 | **0.586** | **0.470** | 0.598 | **0.772** | 0.683 |
| bm25_tuned+ppr0.1_dev | 0.265 | **0.586** | 0.416 | 0.586 | 0.700 | 0.696 |
| bm25_tuned+trust0.2_dev | 0.257 | 0.572 | 0.428 | 0.590 | 0.711 | 0.665 |
| bm25_tuned_dev | 0.265 | **0.586** | 0.441 | **0.600** | 0.743 | 0.696 |
| bm25f_tuned_dev | 0.143† | 0.269 | 0.235 | 0.330 | 0.495 | 0.326 |
| tfidf_lnc.ltc_dev (baseline) | **0.317** | 0.521 | 0.374 | 0.522 | 0.596 | **0.796** |

† p < 0.05 vs. tfidf_lnc.ltc_dev (paired randomization test, 10000 permutations, Holm-corrected). n = 23 queries.
