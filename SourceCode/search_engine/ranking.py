import math
from collections import Counter

import numpy as np

from common import tokenize


def _topk(ids, vals, k):
    """Top-k (doc_id, score) pairs, score descending, ties broken by smaller doc id."""
    if k <= 0 or len(vals) == 0:
        return []
    if len(vals) > k:
        kth = np.partition(vals, len(vals) - k)[len(vals) - k]
        keep = vals >= kth  # keep every doc tied at the cutoff so the tie-break below is deterministic
        ids, vals = ids[keep], vals[keep]
    order = np.lexsort((ids, -vals))[:k]
    return [(int(ids[i]), float(vals[i])) for i in order]


def tfidf_cosine(idx, query, k=10):
    """lnc.ltc (SMART, document.query): document = (1+log tf), no idf, cosine; query = (1+log tf) x log(N/df), cosine."""
    q_tf = Counter(t for t in tokenize(query) if idx.df(t))
    if not q_tf:
        return []
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
    """Okapi BM25, Lucene-style non-negative idf = ln(1 + (N-df+0.5)/(df+0.5)); global avgdl."""
    scores = np.zeros(idx.N)
    for t in dict.fromkeys(tokenize(query)):  # dedupe, keeping order so float sums are reproducible across runs
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
