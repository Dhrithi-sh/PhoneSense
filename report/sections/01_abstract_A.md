# 1. Abstract

> Report section (owner A). Numbers from `results/ablation_test.md`, `results/kappa.json` and
> Section 8.5.

PhoneSense is a search engine that turns a free-text shopping need ("5G phone under $300 with
a big battery") into a ranked list of phones, built on 7,166 phones and 575,816 reviews from
the Amazon Reviews 2023 Cell Phones data. It combines classic retrieval (a positional inverted
index, TF-IDF, BM25 and a field-weighted variant, Boolean spec filters and phrase search),
neural retrieval (dense retrieval, Rocchio feedback, LLM-based HyDE expansion and rank fusion)
and graph-based reputation (PageRank over a co-review graph and a reviewer-trust score), and
tests the reputation methods against simulated fake reviews. We built a test collection of 45
queries (27 dev, 18 test) with graded labels from an LLM judge, whose agreement with our own
labels is fair (Cohen's $\kappa = 0.26$). All parameters were tuned on dev only. On the test
queries, the system chosen on dev, reciprocal rank fusion of tuned BM25 and dense retrieval,
reaches nDCG@10 0.677 against 0.663 for tuned BM25; BM25 with spec filters scores highest on
test (0.693) but lost on dev. With 16 scored test queries, no system is significantly better
than tuned BM25 after Holm correction, and the only significant difference is a loss
(field-weighted BM25). Graph and trust priors change test nDCG@10 by at most 0.011. Against
naive fake reviews the trust score reduces rating shifts somewhat, but fake accounts that first
copy the consensus defeat it. The main lesson is that, on this collection, a well-tuned lexical
baseline is hard to beat, and the shallow judging pool and missing specifications (77% of
phones have no price) limit what the evaluation can show.
