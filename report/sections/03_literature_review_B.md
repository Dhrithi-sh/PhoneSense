# 2. Literature Review

> **Status: outline (owner: B).** The seven papers below are the planned core set. Before the
> final report, open each one and check authors, year, venue and pages; every `verify` mark
> must be cleared. Citations are written from memory and must not be trusted until checked.

## 2.1 Planned structure

1. **Lexical retrieval**: what our baselines are (TF-IDF lnc.ltc from the textbook, BM25).
2. **Query refinement**: Rocchio feedback, and why pseudo-feedback can drift.
3. **Neural and LLM-assisted retrieval**: sentence embeddings, HyDE.
4. **Fusion**: RRF.
5. **Graph and reputation signals**: PageRank (detail belongs to C's Methodology).
6. **Evaluation with LLM judges**: how far LLM relevance labels can be trusted.
7. **Gap**: no work we found combines these for product search over spec-heavy, review-rich
   items; state our contribution against the table in 2.3.

## 2.2 Core papers (5-7 required)

| # | Paper (verify each) | Used in PhoneSense for | What to say (1-2 sentences) |
|---|---|---|---|
| 1 | Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, Foundations and Trends in IR, 2009 (verify) | BM25 baseline (T05) | Term-frequency saturation (k1) and length normalisation (b) are why BM25 beats raw TF-IDF on short, uneven documents such as listings. |
| 2 | Rocchio, *Relevance Feedback in Information Retrieval*, in *The SMART Retrieval System*, 1971 (verify) | Rocchio feedback (T14) | Moves the query towards judged-relevant documents. We use pseudo-relevance (top-m), so the risk is drift when the top results are wrong. |
| 3 | Reimers & Gurevych, *Sentence-BERT*, EMNLP-IJCNLP 2019 (verify) | Dense retrieval (T12) | Sentence embeddings compared by cosine make semantic search cheap; our encoder is a SBERT-family model. |
| 4 | Gao, Ma, Lin & Callan, *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE), ACL 2023, arXiv 2212.10496 (verify) | HyDE expansion (T15) | An LLM writes a hypothetical answer document; searching with it bridges the vocabulary gap. We adapt it to sparse weights through Rocchio. |
| 5 | Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009 (verify) | Rank fusion (T13) | Scores of different systems are not comparable, ranks are; k = 60 is the published default we start from and tune on dev. |
| 6 | Page, Brin, Motwani & Winograd, *The PageRank Citation Ranking*, Stanford Tech. Report, 1999 (verify) | Graph score (C, T17) | Importance by random-walk on a link graph, with damping and dangling-node handling; we run it on the bought-together graph. |
| 7 | Thomas, Spielman, Craswell & Mitra, *Large Language Models can Accurately Predict Searcher Preferences*, SIGIR 2024, arXiv 2309.10621 (verify) | LLM judge (T09) | LLM labels can approach human labels on a good prompt, but need a human check; we report kappa on a shared sample. |

**Textbook (cite, not counted):** Manning, Raghavan & Schuetze, *Introduction to Information
Retrieval*, Cambridge University Press, 2008 (verify): lnc.ltc, pooling, MAP, nDCG, Rocchio.

**Optional if space allows:** Kleinberg, *Authoritative Sources in a Hyperlinked Environment*,
JACM 1999 (verify), for the HITS idea behind the reviewer-trust score (C); Karpukhin et al.,
*Dense Passage Retrieval for Open-Domain Question Answering*, EMNLP 2020 (verify), for
trained dense retrievers versus our off-the-shelf encoder; Voorhees, *Variations in relevance
judgments and the measurement of retrieval effectiveness*, IPM 2000 (verify), for judge
disagreement not changing system rankings.

## 2.3 Comparison table to fill

| Work | Lexical | Dense | Feedback / expansion | Fusion | Graph / trust | Attack study | Product spec filters |
|---|---|---|---|---|---|---|---|
| BM25 (1) | yes | | | | | | |
| Rocchio (2) | | | yes | | | | |
| SBERT (3) | | yes | | | | | |
| HyDE (4) | | yes | yes | | | | |
| RRF (5) | | | | yes | | | |
| PageRank (6) | | | | | yes | | |
| **PhoneSense** | yes | yes | yes | yes | yes | yes | yes |

## 2.4 To do before the final report

- [ ] Verify all citation details against the papers themselves.
- [ ] Write 2-3 sentences per paper in the voice of 2.2, then merge into flowing paragraphs.
- [ ] Add what each paper does *not* do (the gap) in one clause.
- [ ] Format references as the course requires and send them to C for `10_references_C.md`.
