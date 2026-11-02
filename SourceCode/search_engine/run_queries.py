"""Run diverse queries on TF-IDF and BM25, print top results, overlap, and a k1/b sweep.

With --out DIR the same results are also written as CSV (summary.csv, top_results.csv, bm25_sweep.csv)
so tables in the report come from files.
"""
import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_engine.index import InvertedIndex
from search_engine.ranking import bm25, tfidf_cosine

QUERIES = [
    "cockroach",                         # single rare term, restaurants and hotels
    "amazing pizza",                     # food, short, common terms
    "friendly staff but slow service",   # long phrase with a near-stopword
    "gluten-free options",               # hyphenated token (tokenizer-dependent)
    "oil change",                        # automotive
    "hotel room clean quiet",            # travel, topical
    "rude receptionist long wait",       # medical / services
]


def snippet(idx, d, n=110):
    r = idx.doc(d)
    t = r["text"].replace("\n", " ")
    return f"[{r['name']}, {r['city']}] " + t[:n] + ("..." if len(t) > n else "")


RUNS = 5


def timed(fn, *args):
    """(result, median milliseconds over RUNS calls); each call allocates an N-sized score array."""
    ms = []
    for _ in range(RUNS):
        t0 = time.perf_counter(); res = fn(*args); ms.append(1e3 * (time.perf_counter() - t0))
    return res, statistics.median(ms)


def overlap(a, b):
    sa, sb = {d for d, _ in a}, {d for d, _ in b}
    return len(sa & sb) / max(1, len(sa | sb))


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", help="index directory built by search_engine.index")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--queries", nargs="+", default=QUERIES)
    ap.add_argument("--out", help="directory to write summary.csv, top_results.csv and bm25_sweep.csv")
    a = ap.parse_args()
    idx = InvertedIndex.load(a.index)
    print(f"N={idx.N} avgdl={idx.avgdl:.1f}\n")
    depth = max(10, a.k)  # results kept per ranker; Jaccard is computed over this depth
    tfidf_cosine(idx, a.queries[0], depth); bm25(idx, a.queries[0], depth)  # warm the disk cache so timings are fair
    summary, top = [], []
    for q in a.queries:
        (tf, ms_tf), (bm, ms_bm) = timed(tfidf_cosine, idx, q, depth), timed(bm25, idx, q, depth)
        jac = overlap(tf, bm)
        print(f"=== {q!r}  (tfidf {ms_tf:.1f}ms, bm25 {ms_bm:.1f}ms median of {RUNS}, Jaccard@{depth}={jac:.2f})")
        summary.append([q, idx.N, depth, f"{ms_tf:.2f}", f"{ms_bm:.2f}", f"{jac:.3f}"])
        for name, res in (("TF-IDF", tf), ("BM25", bm)):
            print(f" [{name}]")
            for rank, (d, s) in enumerate(res, 1):
                if rank <= a.k:
                    print(f"   {s:7.3f} len={idx.doc_len[d]:4d} {snippet(idx, d)}")
                if a.out:
                    r = idx.doc(d)
                    top.append([q, name, rank, d, f"{s:.6f}", int(idx.doc_len[d]), r["review_id"], r["name"],
                                r["stars"], r["text"].replace("\n", " ")])
        print()
    q = max(a.queries, key=lambda x: len(x.split()))  # sweep on the longest query
    print(f"--- BM25 parameter sweep for {q!r}: mean length (tokens) of top-10 docs")
    sweep = []
    for k1 in (0.5, 1.2, 2.0):
        for b in (0.0, 0.75, 1.0):
            res = bm25(idx, q, 10, k1, b)
            mean_len = sum(idx.doc_len[d] for d, _ in res) / max(1, len(res))
            print(f"   k1={k1} b={b}: {mean_len:.1f}")
            sweep.append([q, k1, b, f"{mean_len:.1f}"])
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        write_csv(os.path.join(a.out, "summary.csv"), ["query", "N", "depth", "tfidf_ms", "bm25_ms", "jaccard"], summary)
        write_csv(os.path.join(a.out, "top_results.csv"),
                  ["query", "ranker", "rank", "doc_id", "score", "doc_len", "review_id", "business", "stars", "text"], top)
        write_csv(os.path.join(a.out, "bm25_sweep.csv"), ["query", "k1", "b", "mean_top10_len"], sweep)
        print(f"\nwrote CSVs to {a.out}")
    idx.close()


if __name__ == "__main__":
    main()
