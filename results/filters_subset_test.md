| Run | Subset | n | P@10 | MAP | nDCG@10 | judged@10 | p (nDCG@10 vs base) |
|---|---|---:|---:|---:|---:|---:|---:|
| bm25_tuned_test | with constraints | 4 | 0.375 | 0.575 | 0.704 | 0.625 | - |
| bm25_tuned_test | without constraints | 12 | 0.383 | 0.596 | 0.649 | 0.617 | - |
| bm25_tuned+filters_test | with constraints | 4 | 0.450 | 0.706 | 0.825 | 0.450 | 0.1203 |
| bm25_tuned+filters_test | without constraints | 12 | 0.383 | 0.596 | 0.649 | 0.617 | 1.0000 |
| bm25_tuned+phrase_test | with constraints | 4 | 0.375 | 0.575 | 0.704 | 0.625 | 1.0000 |
| bm25_tuned+phrase_test | without constraints | 12 | 0.383 | 0.596 | 0.649 | 0.617 | 1.0000 |
| bm25_tuned+filters+phrase_test | with constraints | 4 | 0.450 | 0.706 | 0.825 | 0.450 | 0.1203 |
| bm25_tuned+filters+phrase_test | without constraints | 12 | 0.383 | 0.596 | 0.649 | 0.617 | 1.0000 |
