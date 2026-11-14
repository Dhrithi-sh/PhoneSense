# Sample queries and judgments

Small made-up test data for the 20 articles in the sample index, so the
evaluation code can run end to end before the real crawl and judging are done.
These are not our real results.

- `queries.tsv`: one query per line: `qid`, `need_id`, `form` (hindi, hinglish, english), `query`.
  Several queries can share one information need, written in different forms.
- `qrels.txt`: judgments per information need: `need_id 0 doc_id grade`, grade 0, 1 or 2.

The Hinglish and English queries return nothing yet, because the sample setup
only does exact matching. They start working once the Hinglish layer and the
cross-lingual layer are plugged in.
