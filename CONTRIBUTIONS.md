# CONTRIBUTIONS

Fill at Gate 4. Numbers come from the task board ("Done by" and "WU done") and are checked against
`git shortlog -sn --no-merges` and the merged PRs. If the real split is uneven, write the real
split. Each member fills **their own row**; the WU column is the planned WU of the tasks counted as
completed (a task counts only when its code, tests and run files exist and are pushed).

| Member | Name | GitHub | Completed WU | Task IDs | Report sections written |
|---|---|---|---|---|---|
| A | | | | | |
| B | Palleti Viraja | virajapalleti | 30.5 (details below) | T08, T12, T13, T14, T15, T09a, T09b, L-B, T23; T24 in progress | Introduction, Literature Review, Objective, Proposed Model (drafts: result numbers still to fill) |
| C | | | | | |

## Member B: details

| Task | Planned WU | Status | Where |
|---|---|---|---|
| T08 45 queries, dev/test split | 4 | done, merged (PR #2) | `queries/queries.jsonl` |
| T12 dense retrieval with embedding cache | 4 | done, PR #3 | `phonesense/retrieval/dense.py` |
| T14 Rocchio pseudo-relevance feedback | 3 | done, PR #4; tuned on dev | `phonesense/retrieval/rocchio.py` |
| T15 HyDE expansion (every LLM call logged) | 5 | done, tuned on dev | `phonesense/retrieval/hyde.py`, `phonesense/llm/` |
| T09a LLM judge and pooling | 3 | done; round 1 on dev (495 pairs) | `phonesense/llm/judge.py`, `pooling.py` |
| T09b round 2 on test (305 pairs), kappa | 2 | done; kappa on 89 human labels | PR #6, `results/kappa.json` |
| L-B hand-labelling share | 1.5 | done (30 pairs) | `qrels/human.txt` |
| T13 fusion: RRF and weighted linear, tuned on dev | 4 | done; fused runs written | `phonesense/retrieval/fusion.py`, `scripts/run_fusion.py` |
| T23 buying-guide generator | 4 | done (figures checked against the data; wrong attribution of a real figure is not caught) | `phonesense/llm/buying_guide.py` |
| T24 demo UI | 8 | in progress: tabs, system toggles, score breakdown, priors, guide button and the attack tab are built; Boolean filters and phrase search are not wired yet | `phonesense/ui/` |
| R-B1 Introduction and literature review | 5 | drafted (six core citations checked, two not re-checked); result numbers to fill | `report/sections/` |
| R-B2 Objective and proposed model | 2 | drafted with a pipeline figure; result numbers to fill | `report/sections/`, `report/figures/pipeline.png` |

Work beyond the plan: an LLM client with Gemini, OpenAI, Mistral and Groq backends (logging,
replay, throttling), the pooling and labelling pipeline (adaptive re-pooling of queries without a
relevant phone, sheets and a terminal helper for hand labels), the LLM-vs-human agreement
analysis and a label-sensitivity check, and tuning of Rocchio, HyDE and fusion on dev only.
