# 3.3 Search Engine: Inside the Ranking Black Box

Draft of our section for the group report. The assignment gives no formula for either ranker, so the variants below are our choices. Figures are from the full dataset: 6,990,280 reviews joined to `business.json`, tokenized with the group's custom tokenizer.

## 3.3.1 Indexing

We built the inverted index ourselves (no Lucene for storage). Each term maps to a postings list of (document id, term frequency) pairs, stored on disk as two typed arrays read through `numpy.memmap`: `uint32` document ids and `uint16` term frequencies, six bytes per posting. Term frequency is capped at 65,535, which no review reaches in practice. A vocabulary table gives each term's start offset and document frequency, so a query reads only the postings of its own terms.

Per-document data kept at index time: token count (`doc_len`), the cosine norm used by TF-IDF (`doc_norm`), the byte offset of the review line in `review.json` (review text is re-read from there, not stored), and a pointer to the business record. The **average document length is computed once over the whole collection during indexing** (avgdl = 105.9 tokens) and stored with the index, as the assignment requires.

| | |
|---|---|
| Documents | 6,990,280 |
| Distinct terms | 966,439 |
| Postings | 495,219,157 |
| Index size on disk | 3.0 GB |
| Build time | about 9 minutes (single process) |

An index stores a probe string tokenized at build time; loading fails with an error if the tokenizer has changed since, so queries can never be tokenized differently from the index.

## 3.3.2 Ranking implementation

Notation: N documents, df(t) documents containing term t, tf(t,d) frequency of t in d, dl(d) length of d, avgdl the global average length.

**TF-IDF with cosine similarity: SMART `lnc.ltc`** (the scheme the lecture names as a very standard weighting; document.query)

- document weight: w(t,d) = 1 + log tf(t,d), no idf, divided by the Euclidean norm of the document's weight vector (precomputed in `doc_norm`);
- query weight: w(t,q) = (1 + log tf(t,q)) · log(N / df(t)), divided by the query vector's norm;
- score(q,d) = Σ_t w(t,q) · w(t,d) / (‖q‖ · ‖d‖).

**Okapi BM25**

score(q,d) = Σ_{t∈q} idf(t) · tf(t,d)(k1+1) / ( tf(t,d) + k1(1 − b + b·dl(d)/avgdl) ),  idf(t) = ln(1 + (N − df + 0.5)/(df + 0.5))

with k1 = 1.2, b = 0.75 by default (both are parameters of the function). This is the original Okapi form with the (k1+1) numerator; the non-negative idf avoids negative scores for terms occurring in more than half of the documents. Repeated query terms are counted once.

Core of the two scorers (`search_engine/ranking.py`; scores are accumulated term by term over postings only, then the top k is selected with a deterministic tie-break on document id):

```python
def tfidf_cosine(idx, query, k=10):
    q_tf = Counter(t for t in tokenize(query) if idx.df(t))
    scores, q_sq = np.zeros(idx.N), 0.0
    for t, qtf in q_tf.items():
        idf = math.log(idx.N / idx.df(t))
        qw = (1 + math.log(qtf)) * idf
        q_sq += qw * qw
        ids, tfs = idx.postings(t)
        scores[ids] += qw * (1 + np.log(tfs.astype(np.float64)))
    cand = np.flatnonzero(scores)
    return _topk(cand, scores[cand] / ((math.sqrt(q_sq) or 1.0) * idx.doc_norm[cand]), k)

def bm25(idx, query, k=10, k1=1.2, b=0.75):
    scores = np.zeros(idx.N)
    for t in dict.fromkeys(tokenize(query)):
        df = idx.df(t)
        if not df:
            continue
        idf = math.log(1 + (idx.N - df + 0.5) / (df + 0.5))
        ids, tfs = idx.postings(t)
        tfs = tfs.astype(np.float64)
        norm = k1 * (1 - b + b * idx.doc_len[ids] / idx.avgdl)
        scores[ids] += idf * tfs * (k1 + 1) / (tfs + norm)
    cand = np.flatnonzero(scores)
    return _topk(cand, scores[cand], k)
```

Correctness: our BM25 is cross-checked against Lucene's BM25 on the same tokens (see the comparison below); scores differ only by the constant factor explained there.

## Analysis

**Queries.** Seven queries chosen to differ in type and domain. The dataset is not only restaurants (about 43% of businesses are in Restaurants/Food), so the set spans food, automotive, travel and medical/service reviews. Latency is the median of five runs; each call allocates an N-sized score array.

| # | Query | Type | df of terms | TF-IDF ms | BM25 ms | top-10 Jaccard |
|---|---|---|---|---|---|---|
| 1 | cockroach | single rare term | 2,233 | 14.8 | 13.6 | 0.54 |
| 2 | amazing pizza | food, two common terms | 759k, 324k | 41.2 | 40.0 | 0.00 |
| 3 | friendly staff but slow service | long phrase, near-stopword | 969k, 997k, 3.16M, 143k, 1.9M | 138.2 | 136.6 | 0.05 |
| 4 | gluten-free options | hyphenated token | 8,275; 269,725 | 23.4 | 21.5 | 0.18 |
| 5 | oil change | automotive | 84.6k, 130k | 22.4 | 20.3 | 0.11 |
| 6 | hotel room clean quiet | travel | 159k, 335k, 417k, 62k | 39.9 | 39.9 | 0.33 |
| 7 | rude receptionist long wait | medical / services | 177k, 16k, 443k, 627k | 42.0 | 42.6 | 0.18 |

**TF-IDF and BM25 rank differently.** Top-10 overlap is low (Jaccard 0.00 to 0.54). They differ in where idf enters, in how term frequency saturates (logarithmic versus the k1 curve) and in length handling (cosine norm versus b).

**TF-IDF is more short-biased than BM25.** For "amazing pizza", TF-IDF ranks the two-token review "Good pizza" second, while BM25's top five are 27 to 64 tokens ("The pizza is always amazing…"). For "oil change", TF-IDF's top five are 7 to 28 tokens and BM25's are 28 to 199. BM25 is still short-biased overall: the mean length of its top 10 at b = 0.75 is 26 tokens against a collection average of 105.9.

**The parameter b controls the length bias; k1 hardly matters here.** On query 3, the mean length of the BM25 top 10 is about 530 tokens with b = 0 (no normalization), 26 with b = 0.75 and 16 to 17 with b = 1. Varying k1 over 0.5, 1.2 and 2.0 leaves these numbers almost unchanged.

| mean top-10 length (tokens) | b = 0 | b = 0.75 | b = 1 |
|---|---|---|---|
| k1 = 0.5 | 531 | 26 | 17 |
| k1 = 1.2 | 540 | 26 | 17 |
| k1 = 2.0 | 523 | 26 | 16 |

**Tokenizer effects.** The tokenizer keeps "gluten-free" as one token (df 8,275), so the query "gluten-free options" matches reviews that write it hyphenated, while "gluten free" would not. A near-stopword in a query ("but", df 3.16M) makes the postings lists long and is the main reason query 3 takes about 140 ms against 14 ms for the single rare term.

**Domain coverage.** Non-food queries return the right kind of business: "oil change" returns auto-service shops, "hotel room clean quiet" hotels, and "rude receptionist long wait" a hotel front desk, a restaurant and a medical lab.

## Comparison with Lucene

We indexed the same tokens with Lucene (Pyserini 1.6.0, whitespace analyzer on our pre-tokenized text), so the two systems see identical terms and differences come from scoring only. Lucene indexed 6,990,251 documents; the other 29 reviews contain no tokens and Lucene skips empty documents.

**BM25 against Lucene's BM25** (same k1 = 1.2, b = 0.75). Top-10 overlap is 10/10 for five queries, 9/10 for "amazing pizza" and 8/10 for "friendly staff but slow service"; the top-1 document agrees on five of seven queries.

| Query | overlap@10 | same order | top-1 same | relative score gap rank 10 to 11 (ours / Lucene) |
|---|---|---|---|---|
| cockroach | 10 | yes | yes | 0.0046 / 0.0007 |
| amazing pizza | 9 | no | yes | 0.0011 / 0.0004 |
| friendly staff but slow service | 8 | no | yes | 0.0000 / 0.0000 |
| gluten-free options | 10 | no | no | 0.0070 / 0.0070 |
| oil change | 10 | no | yes | 0.0069 / 0.0071 |
| hotel room clean quiet | 10 | no | no | 0.0051 / 0.0088 |
| rude receptionist long wait | 10 | yes | yes | 0.0271 / 0.0227 |

Lucene's scores are ours divided by about 2.2 (for example 15.129 against 6.887, and 9.829 against 4.468). Lucene's `BM25Similarity` computes the term-frequency part as freq / (freq + k1(1 − b + b·dl/avgdl)), without the (k1+1) numerator; (k1+1) = 2.2 is a constant, so it does not change the ranking. The ratio is not exactly 2.2 for every document, because Lucene stores document length in one lossy byte. The same effect explains the two top-1 differences: the competing documents are nearly tied (16.967 against 16.964 for "gluten-free options", 22.548 against 22.489 for "hotel room clean quiet"). The overlaps below 10 are cut-off ties as well: for the two queries concerned the relative score gap between rank 10 and rank 11 is 0.0011 and 0.0000, so the documents at the boundary are practically indistinguishable.

**TF-IDF against Lucene's `ClassicSimilarity`.** Top-10 overlap is 7, 5, 6, 8, 8, 9 and 10 (queries 1 to 7). This is a comparison between two variants and not a correctness check: Lucene's classic scoring uses tf = √freq, idf = 1 + log((docCount+1)/(docFreq+1)), a length norm of 1/√length and no cosine over the document vector (Lucene 9.10 javadoc). Because it is a different variant, this comparison does not validate our TF-IDF.

## Limitations

- The assignment specifies no TF-IDF or BM25 variant, so a different variant would rank differently (as the Lucene comparison shows for TF-IDF).
- Ranking quality is judged qualitatively (what each ranker returns, length bias, agreement with Lucene). We did not collect relevance judgments, so there is no precision@k or similar measure.
- Boolean retrieval is not implemented.
- Latencies are single-process timings on a 16 GB machine with the index memory-mapped from local disk.

## Third-party software

NumPy (arrays, memory mapping); Apache Lucene via Pyserini 1.6.0 (baseline only); the group's custom tokenizer (`regex` package).
