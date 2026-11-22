| System | Owner | Group | P@10 | R@10 | MAP | nDCG@10 | MRR | judged@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF lnc.ltc | A | Lexical | 0.375 | 0.590 | 0.485 | 0.544 | 0.673 | **0.781** |
| BM25 (k1=1.2, b=0.75)‡ | A | Lexical | 0.369 | 0.609 | 0.579 | 0.653 | 0.758 | 0.606 |
| BM25 tuned‡ | A | Lexical | 0.381 | 0.634 | 0.591 | 0.663 | 0.755 | 0.619 |
| BM25F tuned‡ | A | Lexical | 0.263† | 0.470† | 0.424† | 0.469† | 0.701 | 0.369 |
| + filters‡ | A | +Structure | **0.400** | **0.653** | **0.624** | **0.693** | **0.805** | 0.575 |
| + phrase‡ | A | +Structure | 0.381 | 0.634 | 0.591 | 0.663 | 0.755 | 0.619 |
| + filters + phrase‡ | A | +Structure | **0.400** | **0.653** | **0.624** | **0.693** | **0.805** | 0.575 |
| Dense | B | Neural | pending: no run (B) | - | - | - | - | - |
| Rocchio PRF | B | Neural | pending: no run (B) | - | - | - | - | - |
| HyDE | B | Neural | pending: no run (B) | - | - | - | - | - |
| RRF (BM25, dense) | B | Fusion | pending: no run (B) | - | - | - | - | - |
| Weighted fusion | B | Fusion | pending: no run (B) | - | - | - | - | - |
| PageRank rerank‡ | C | Graph+Trust | 0.375 | 0.628 | 0.594 | 0.660 | 0.753 | 0.600 |
| Personalized PR‡ | C | Graph+Trust | 0.381 | 0.634 | 0.598 | 0.671 | 0.753 | 0.619 |
| Trust-weighted rerank‡ | C | Graph+Trust | 0.381 | 0.634 | 0.604 | 0.674 | 0.795 | 0.606 |
| Best overall: BM25 tuned‡ | A | Best (chosen on dev) | 0.381 | 0.634 | 0.591 | 0.663 | 0.755 | 0.619 |

Test split (16 queries). †: p < 0.05 vs. BM25 tuned (paired randomization test, 10,000 permutations, Holm-corrected across all rows). ‡: judged@10 < 0.7, i.e. the run was poorly covered by the judging pool and its scores are lower bounds (unjudged documents count as non-relevant). Best value per column in bold. Best overall is chosen by dev nDCG@10.
