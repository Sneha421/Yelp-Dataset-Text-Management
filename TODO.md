# Deferred work (not blocking)

## 3.3 Search engine: analysis tooling (add after the tokenizer lands, or any time before the deadline)
The engine is finished and tested. These make the analysis repeatable; none needs the final tokenizer to be *written*,
only to produce the *final numbers*. Re-run everything after the tokenizer swap (rebuild index + Lucene input).

- [ ] Designed query set in `search_engine/run_queries.py` (assignment: at least 5 diverse queries; we aim for 5 to 8).
      Types: single common term, rare terms, multi-aspect, intent without keyword, long natural-language,
      repeated words, misspelling/out-of-vocabulary, stop-word only. Add a check that each query term has df > 0.
- [ ] Full 7M index build and Lucene input once the tokenizer is final (~4 min / ~4 GB estimated from the 1M run).

## Open decisions
- Query set design.
- Boolean retrieval: decided out of scope.
- Lucene on full data or a stated sample. (Lucene always uses our custom tokenizer via --pretokenized; no default-analyzer run.)

## Optional extras (NOT required by the assignment; only if time allows)
- [ ] Relevance judgments: pool the top-10 of TF-IDF, BM25 (and Lucene) per query into a CSV for relevant/not-relevant marking,
      then a precision@10 script (mirrors `application/evaluate.py`). About 100 results to read. Adds a measured quality result
      on top of the qualitative analysis. Decide who judges.
