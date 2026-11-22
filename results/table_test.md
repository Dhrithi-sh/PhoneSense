| Run | P@10 | R@10 | MAP | nDCG@10 | RR | judged@10 |
|---|---:|---:|---:|---:|---:|---:|
| bm25_default_test | 0.369 | 0.609 | 0.579 | 0.653 | 0.758 | 0.606 |
| bm25_tuned+filters+phrase_test | 0.400 | 0.653 | **0.624** | **0.693** | **0.805** | 0.575 |
| bm25_tuned+filters_test | 0.400 | 0.653 | **0.624** | **0.693** | **0.805** | 0.575 |
| bm25_tuned+pagerank0.1_test | 0.375 | 0.628 | 0.594 | 0.660 | 0.753 | 0.600 |
| bm25_tuned+phrase_test | 0.381 | 0.634 | 0.591 | 0.663 | 0.755 | 0.619 |
| bm25_tuned+ppr0.1_test | 0.381 | 0.634 | 0.598 | 0.671 | 0.753 | 0.619 |
| bm25_tuned+trust0.2_test | 0.381 | 0.634 | 0.604 | 0.674 | 0.795 | 0.606 |
| bm25_tuned_test | 0.381 | 0.634 | 0.591 | 0.663 | 0.755 | 0.619 |
| bm25f_tuned_test | 0.263 | 0.470 | 0.424 | 0.469 | 0.701 | 0.369 |
| fusion_linear_w55-45_bm25_tuned+dense_sentence-transformers-all-minilm-l6-v2_test | 0.394 | 0.711 | 0.582 | 0.681 | 0.802 | 0.775 |
| hyde_tfidf_lnc.ltc_m10_b0_g0_t30_d0.35_test | 0.356 | 0.575 | 0.515 | 0.564 | 0.717 | 0.731 |
| rocchio_tfidf_lnc.ltc_m10_b0.1_g0_t50_test | 0.375 | 0.562 | 0.476 | 0.526 | 0.673 | 0.781 |
| rrf_k5_bm25_tuned+dense_sentence-transformers-all-minilm-l6-v2_test | **0.412** | **0.724** | 0.524 | 0.677 | 0.673 | **0.863** |
| tfidf_lnc.ltc_test (baseline) | 0.375 | 0.590 | 0.485 | 0.544 | 0.673 | 0.781 |

† p < 0.05 vs. tfidf_lnc.ltc_test (paired randomization test, 10000 permutations, Holm-corrected). n = 16 queries.
