| Run | Subset | n | P@10 | MAP | nDCG@10 | judged@10 | p (nDCG@10 vs base) |
|---|---|---:|---:|---:|---:|---:|---:|
| bm25_tuned_dev | with constraints | 9 | 0.222 | 0.407 | 0.519 | 0.700 | - |
| bm25_tuned_dev | without constraints | 14 | 0.293 | 0.463 | 0.653 | 0.693 | - |
| bm25_tuned+filters_dev | with constraints | 9 | 0.178 | 0.256 | 0.422 | 0.311 | 0.2146 |
| bm25_tuned+filters_dev | without constraints | 14 | 0.293 | 0.463 | 0.653 | 0.693 | 1.0000 |
| bm25_tuned+phrase_dev | with constraints | 9 | 0.222 | 0.407 | 0.519 | 0.700 | 1.0000 |
| bm25_tuned+phrase_dev | without constraints | 14 | 0.293 | 0.511 | 0.649 | 0.671 | 0.5094 |
| bm25_tuned+filters+phrase_dev | with constraints | 9 | 0.178 | 0.256 | 0.422 | 0.311 | 0.2146 |
| bm25_tuned+filters+phrase_dev | without constraints | 14 | 0.293 | 0.511 | 0.649 | 0.671 | 0.5094 |
