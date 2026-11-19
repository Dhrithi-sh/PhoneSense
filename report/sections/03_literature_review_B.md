# Literature Review

> **Status: full draft (owner: B).** Citation details marked ✔ were checked against the
> publisher, ACL Anthology, arXiv, dblp or the authors' page in November 2026. Items marked
> **(not re-checked)** are from memory and must be verified before the final report. Hand the
> final list to C for `10_references_C.md` (reference format).

## 1. Lexical retrieval

Term weighting by TF-IDF is the textbook baseline for ranked retrieval; we use the SMART
lnc.ltc variant described by Manning, Raghavan and Schuetze (2008). BM25 refines it with term
frequency saturation and document length normalisation inside the probabilistic relevance
framework (Robertson and Zaragoza, 2009). Both match words, not meaning, which is their strength
on model names ("Pixel 7a") and spec tokens ("5000mAh") and their weakness on a need such as
"battery that lasts two days". We implement both ourselves and use them as the baselines every
other component is compared with.

## 2. Query refinement and expansion

Rocchio (1971) moves the query vector towards documents judged relevant and away from those
judged not relevant. With no user to judge, **pseudo-relevance feedback** assumes the top
results of a first pass are relevant; this helps when the first pass is mostly right and drifts
when it is not. HyDE (Gao, Ma, Lin and Callan, 2023) asks an LLM to write a hypothetical
document that answers the query and searches with it, on the idea that the invented document
may contain false details but sits near real relevant ones. Jedidi and Lin (2025) show that for
sparse retrievers such as BM25, scoring HyDE's expansion terms with a feedback model such as
Rocchio works better than concatenating the generated text to the query (their reported gain is
about 1.4 points, 4.2%, on average). Our HyDE component follows this line, applied to phones;
it is their idea, not ours.

## 3. Dense retrieval

Sentence-BERT (Reimers and Gurevych, 2019) produces sentence embeddings that can be compared
with cosine similarity, which makes semantic search cheap once the documents are embedded. Our
dense retriever uses an off-the-shelf encoder of this family on each phone's text, with no
fine-tuning, so any gain comes from the encoder's general knowledge and not from training on
phone data.

## 4. Rank fusion

Rankers fail on different queries and their scores are not on the same scale. Reciprocal Rank
Fusion (Cormack, Clarke and Buettcher, 2009) merges rankings using only ranks, giving a
document `1 / (k + rank)` from each system; the authors report that it outperforms Condorcet
fusion and individual rank-learning methods. We use RRF with the published constant `k = 60` as
the default, and compare it with weighted linear fusion of min-max normalised scores tuned on
development queries.

## 5. Graph and reputation signals

PageRank (Page, Brin, Motwani and Winograd, 1999) scores a node by the stationary behaviour of a
random surfer on a link graph, with damping and a teleport step so that dead ends and cycles do
not break it. We apply it to a "bought together" phone graph and use it as a ranking prior. Star
ratings are a second signal that is easy to manipulate with a few fake reviews; our
reviewer-trust score lets phone quality and reviewer reliability reinforce each other, in the
spirit of hubs and authorities (Kleinberg, 1999, **not re-checked**). Its design and the attack
simulation belong to the Methodology and Experiments sections.

## 6. Evaluating with LLM judges

Graded relevance labels for every pooled (query, phone) pair are too many to write by hand.
Thomas, Spielman, Craswell and Mitra (2024) report that, with a prompt chosen against feedback
from real searchers, an LLM can label as accurately as third-party human labellers and is as
useful for finding the best systems and the hardest queries. We follow the practice, not the
claim: our judge is checked against a human-labelled sample with Cohen's kappa, and we report
the agreement rather than assuming it. Pooling (judging only the union of the systems' top
results) means a phone no system retrieves is never judged and counts as not relevant, which
favours the pooled systems.

## 7. Gap and our position

| Work | Lexical | Dense | Feedback / expansion | Fusion | Graph / trust | Attack study | Spec filters |
|---|---|---|---|---|---|---|---|
| BM25 (Robertson and Zaragoza) | yes | | | | | | |
| Rocchio (1971) | | | yes | | | | |
| Sentence-BERT | | yes | | | | | |
| HyDE; Jedidi and Lin | | yes / sparse | yes | | | | |
| RRF | | | | yes | | | |
| PageRank | | | | | yes | | |
| **PhoneSense** | yes | yes | yes | yes | yes | yes | yes |

Each ingredient is known. What we add is a single, controlled comparison of them on a
spec-heavy, review-rich product domain, with graded labels whose quality we measure, a trust
signal tested under simulated review attacks, and a demo that exposes every component. We do not
claim a new retrieval model.

## References (to pass to C)

1. Manning, C. D., Raghavan, P. and Schuetze, H. (2008). *Introduction to Information Retrieval*.
   Cambridge University Press. **(not re-checked)**
2. Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and
   Beyond. *Foundations and Trends in Information Retrieval* 3(4), 333-389.
   doi:10.1561/1500000019. ✔
3. Rocchio, J. J. (1971). Relevance feedback in information retrieval. In G. Salton (ed.), *The
   SMART Retrieval System: Experiments in Automatic Document Processing*, Prentice-Hall,
   313-323. ✔
4. Gao, L., Ma, X., Lin, J. and Callan, J. (2023). Precise Zero-Shot Dense Retrieval without
   Relevance Labels. *Proc. ACL 2023* (long papers), paper 2023.acl-long.99; arXiv:2212.10496.
   ✔ (page numbers not checked)
5. Jedidi, N. and Lin, J. (2025). Revisiting Feedback Models for HyDE. arXiv:2511.19349. ✔
   (published venue not confirmed; one page lists SIGIR 2026)
6. Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese
   BERT-Networks. *Proc. EMNLP-IJCNLP 2019*, 3982-3992. doi:10.18653/v1/D19-1410. ✔
7. Cormack, G. V., Clarke, C. L. A. and Buettcher, S. (2009). Reciprocal rank fusion outperforms
   Condorcet and individual rank learning methods. *Proc. SIGIR 2009*, 758-759.
   doi:10.1145/1571941.1572114. ✔
8. Page, L., Brin, S., Motwani, R. and Winograd, T. (1999). The PageRank Citation Ranking:
   Bringing Order to the Web. Stanford InfoLab Technical Report 1999-66. ✔
9. Kleinberg, J. M. (1999). Authoritative sources in a hyperlinked environment. *Journal of the
   ACM* 46(5), 604-632. **(not re-checked)**
10. Thomas, P., Spielman, S., Craswell, N. and Mitra, B. (2024). Large Language Models Can
    Accurately Predict Searcher Preferences. *Proc. SIGIR 2024*; arXiv:2309.10621. ✔ (page
    numbers not checked)
