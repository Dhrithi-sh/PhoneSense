| System | Owner | Group | P@10 | R@10 | MAP | nDCG@10 | MRR | judged@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF lnc.ltc | A | Lexical | 0.375 | 0.590 | 0.485 | 0.544 | 0.673 | 0.781 |
| BM25 (k1=1.2, b=0.75)‡ | A | Lexical | 0.369 | 0.609 | 0.579 | 0.653 | 0.758 | 0.606 |
| BM25 tuned‡ | A | Lexical | 0.381 | 0.634 | 0.591 | 0.663 | 0.755 | 0.619 |
| BM25F tuned‡ | A | Lexical | 0.263† | 0.470† | 0.424 | 0.469† | 0.701 | 0.369 |
| + filters‡ | A | +Structure | 0.400 | 0.653 | **0.624** | **0.693** | **0.805** | 0.575 |
| + phrase‡ | A | +Structure | 0.381 | 0.634 | 0.591 | 0.663 | 0.755 | 0.619 |
| + filters + phrase‡ | A | +Structure | 0.400 | 0.653 | **0.624** | **0.693** | **0.805** | 0.575 |
| Dense‡ | B | Neural | 0.194 | 0.380 | 0.291 | 0.417 | 0.467 | 0.544 |
| Rocchio PRF | B | Neural | 0.375 | 0.562 | 0.476 | 0.526 | 0.673 | 0.781 |
| HyDE | B | Neural | 0.356 | 0.575 | 0.515 | 0.564 | 0.717 | 0.731 |
| RRF (BM25, dense) | B | Fusion | **0.412** | **0.724** | 0.524 | 0.677 | 0.673 | **0.863** |
| Weighted fusion | B | Fusion | 0.394 | 0.711 | 0.582 | 0.681 | 0.802 | 0.775 |
| PageRank rerank‡ | C | Graph+Trust | 0.375 | 0.628 | 0.594 | 0.660 | 0.753 | 0.600 |
| Personalized PR‡ | C | Graph+Trust | 0.381 | 0.634 | 0.598 | 0.671 | 0.753 | 0.619 |
| Trust-weighted rerank‡ | C | Graph+Trust | 0.381 | 0.634 | 0.604 | 0.674 | 0.795 | 0.606 |
| Best overall: RRF (BM25, dense) | B | Best (chosen on dev) | **0.412** | **0.724** | 0.524 | 0.677 | 0.673 | **0.863** |

Test split (16 queries). †: p < 0.05 vs. BM25 tuned (paired randomization test, 10,000 permutations, Holm-corrected across all rows). ‡: judged@10 < 0.7, i.e. the run was poorly covered by the judging pool and its scores are lower bounds (unjudged documents count as non-relevant). Best value per column in bold. Best overall is chosen by dev nDCG@10.
