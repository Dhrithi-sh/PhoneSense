| System | Owner | Group | P@10 | R@10 | MAP | nDCG@10 | MRR | judged@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF lnc.ltc | A | Lexical | 0.317 | 0.521 | 0.374 | 0.522 | 0.596 | 0.796 |
| BM25 (k1=1.2, b=0.75)‡ | A | Lexical | 0.265 | 0.574 | 0.427 | 0.588 | 0.714 | 0.648 |
| BM25 tuned‡ | A | Lexical | 0.265 | 0.586 | 0.441 | 0.600 | 0.743 | 0.696 |
| BM25F tuned‡ | A | Lexical | 0.143† | 0.269† | 0.235† | 0.330† | 0.495 | 0.326 |
| + filters‡ | A | +Structure | 0.248 | 0.548 | 0.382 | 0.562 | 0.706 | 0.543 |
| + phrase‡ | A | +Structure | 0.265 | 0.586 | 0.470 | 0.598 | **0.772** | 0.683 |
| + filters + phrase‡ | A | +Structure | 0.248 | 0.548 | 0.411 | 0.560 | 0.735 | 0.530 |
| Dense | B | Neural | pending: no run (B) | - | - | - | - | - |
| Rocchio PRF | B | Neural | 0.313 | 0.485 | 0.370 | 0.507 | 0.589 | 0.787 |
| HyDE | B | Neural | **0.352** | 0.612 | 0.469 | 0.586 | 0.722 | 0.757 |
| RRF (BM25, dense) | B | Fusion | 0.330 | **0.620** | 0.477 | **0.642** | 0.640 | **0.909** |
| Weighted fusion | B | Fusion | 0.317 | 0.601 | **0.502** | 0.621 | 0.640 | 0.796 |
| PageRank rerank‡ | C | Graph+Trust | 0.248 | 0.566 | 0.434 | 0.582 | 0.739 | 0.648 |
| Personalized PR‡ | C | Graph+Trust | 0.265 | 0.586 | 0.416 | 0.586 | 0.700 | 0.696 |
| Trust-weighted rerank‡ | C | Graph+Trust | 0.257 | 0.572 | 0.428 | 0.590 | 0.711 | 0.665 |
| Best overall: RRF (BM25, dense) | B | Best (chosen on dev) | 0.330 | **0.620** | 0.477 | **0.642** | 0.640 | **0.909** |

Dev split (23 queries). †: p < 0.05 vs. BM25 tuned (paired randomization test, 10,000 permutations, Holm-corrected across all rows). ‡: judged@10 < 0.7, i.e. the run was poorly covered by the judging pool and its scores are lower bounds (unjudged documents count as non-relevant). Best value per column in bold. Best overall is chosen by dev nDCG@10.
