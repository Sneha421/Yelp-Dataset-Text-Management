"""Inverted index over review.json, stored on disk as typed arrays (fits all 7M reviews in 16 GB).

Layout of an index directory:
  doc_ids.u32 / tfs.u16   all postings back to back, term by term, doc ids ascending
  doc_len.npy             tokens per review          offsets.npy   byte offset of each review line
  doc_norm.npy            L2 norm of each review's log-tf vector (for lnc cosine)
  biz.npy                 index of each review's business in meta["businesses"]
  meta.pkl                N, avgdl, vocab (term -> row), per-row postings start/count, business table
Review text is not stored; it is re-read from review.json by byte offset when a result is shown.
"""
import json
import math
import os
import pickle
import sys
from array import array

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_business_info, tokenize

assert (array("I").itemsize, array("H").itemsize, array("Q").itemsize, array("i").itemsize) == (4, 2, 8, 4), "unexpected array item sizes"

TF_MAX = 65535  # postings store tf as uint16
# Tokenizing this at build time and at load time must agree, or queries would use different tokens than the index.
PROBE = "Don't miss the-best-food!! Loved it \U0001F60D #5stars $12.50 caf\u00e9"


class InvertedIndex:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            m = pickle.load(f)
        self.N, self.avgdl = m["N"], m["avgdl"]
        if m.get("probe") != tokenize(PROBE):
            raise RuntimeError("tokenizer changed since this index was built; rebuild it with search_engine.index")
        self.review_path = m["review_path"]
        self.vocab, self.starts, self.counts = m["vocab"], m["starts"], m["counts"]
        self.businesses = m["businesses"]  # list of (business_id, info dict)
        self.doc_len = np.load(os.path.join(path, "doc_len.npy"))
        self.doc_norm = np.load(os.path.join(path, "doc_norm.npy"))
        self.offsets = np.load(os.path.join(path, "offsets.npy"))
        self.biz = np.load(os.path.join(path, "biz.npy"))
        self._ids = np.memmap(os.path.join(path, "doc_ids.u32"), dtype=np.uint32, mode="r")
        self._tfs = np.memmap(os.path.join(path, "tfs.u16"), dtype=np.uint16, mode="r")
        self._fh = None

    @staticmethod
    def load(path):
        return InvertedIndex(path)

    def df(self, term):
        row = self.vocab.get(term)
        return 0 if row is None else int(self.counts[row])

    def postings(self, term):
        """(doc_ids, tfs) arrays for a term; empty if unseen."""
        row = self.vocab.get(term)
        if row is None:
            return self._ids[:0], self._tfs[:0]
        s, n = int(self.starts[row]), int(self.counts[row])
        return self._ids[s:s + n], self._tfs[s:s + n]

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def doc(self, d):
        """Re-read one review (and its business) from review.json."""
        if self._fh is None:
            self._fh = open(self.review_path, "rb")
        self._fh.seek(int(self.offsets[d]))
        r = json.loads(self._fh.readline())
        b = self.businesses[self.biz[d]][1] if self.biz[d] >= 0 else {}
        return {"review_id": r["review_id"], "business_id": r["business_id"], "stars": r["stars"],
                "text": r["text"], "name": b.get("name"), "city": b.get("city"),
                "categories": b.get("categories", "")}


def build_index(review_path, business_path, out_dir, limit=None):
    review_path = os.path.abspath(review_path)
    biz_info = load_business_info(business_path)
    businesses = list(biz_info.items())
    biz_row = {bid: i for i, (bid, _) in enumerate(businesses)}

    vocab, post = {}, []  # term -> row; row -> (array of doc ids, array of tfs)
    doc_len, offsets, biz = array("I"), array("Q"), array("i")
    offset = 0
    with open(review_path, "rb") as f:
        for d, line in enumerate(f):
            if limit is not None and d >= limit:
                break
            r = json.loads(line)
            offsets.append(offset)
            offset += len(line)
            biz.append(biz_row.get(r["business_id"], -1))
            tokens = tokenize(r["text"])
            doc_len.append(len(tokens))
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                row = vocab.get(t)
                if row is None:
                    row = vocab[t] = len(post)
                    post.append((array("I"), array("H")))
                post[row][0].append(d)
                post[row][1].append(min(c, TF_MAX))
    N = len(doc_len)
    if not N:
        raise ValueError("no documents indexed")
    avgdl = sum(doc_len) / N  # global average document length

    os.makedirs(out_dir, exist_ok=True)
    sq = np.zeros(N)
    starts, counts = np.zeros(len(post), dtype=np.int64), np.zeros(len(post), dtype=np.uint32)
    pos = 0
    with open(os.path.join(out_dir, "doc_ids.u32"), "wb") as fi, open(os.path.join(out_dir, "tfs.u16"), "wb") as ft:
        for row, (ids, tfs) in enumerate(post):
            starts[row], counts[row] = pos, len(ids)
            pos += len(ids)
            ids.tofile(fi)
            tfs.tofile(ft)
            w = 1 + np.log(np.frombuffer(tfs, dtype=np.uint16).astype(np.float64))
            sq[np.frombuffer(ids, dtype=np.uint32)] += w * w  # doc ids are unique within a term
    doc_norm = np.sqrt(sq)
    doc_norm[doc_norm == 0] = 1.0

    np.save(os.path.join(out_dir, "doc_len.npy"), np.frombuffer(doc_len, dtype=np.uint32))
    np.save(os.path.join(out_dir, "doc_norm.npy"), doc_norm)
    np.save(os.path.join(out_dir, "offsets.npy"), np.frombuffer(offsets, dtype=np.uint64))
    np.save(os.path.join(out_dir, "biz.npy"), np.frombuffer(biz, dtype=np.int32))
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"N": N, "avgdl": avgdl, "probe": tokenize(PROBE), "review_path": review_path, "vocab": vocab,
                     "starts": starts, "counts": counts, "businesses": businesses},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    return InvertedIndex(out_dir)


if __name__ == "__main__":
    import argparse

    from search_engine.index import build_index  # not the __main__ copy, so pickled data stays importable

    ap = argparse.ArgumentParser(description="Build inverted index directory from review.json")
    ap.add_argument("reviews")
    ap.add_argument("--business", help="business.json (default: next to review.json)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="index")
    a = ap.parse_args()
    business = a.business or os.path.join(os.path.dirname(os.path.abspath(a.reviews)), "business.json")
    idx = build_index(a.reviews, business, a.out, a.limit)
    print(f"docs={idx.N} terms={len(idx.vocab)} postings={len(idx._ids)} avgdl={idx.avgdl:.1f} -> {a.out}")
