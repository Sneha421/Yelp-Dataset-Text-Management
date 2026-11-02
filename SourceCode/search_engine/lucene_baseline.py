"""Lucene (via Pyserini) BM25 baseline for comparison with our own rankers (3.3 analysis). Needs Java 21.

Run with the Pyserini virtualenv from SourceCode/ (see README):
  python -m search_engine.lucene_baseline prepare ../data/review.json --limit 200000 --out ../results/lucene_docs
  python -m search_engine.lucene_baseline index ../results/lucene_docs --out ../results/lucene_index
  python -m search_engine.lucene_baseline compare ../results/lucene_index ../results/index_200k

Reviews are tokenized with our own tokenize() and Lucene reads them with a whitespace analyzer
(--pretokenized), so both engines see identical terms. Lucene doc id = our doc id (line number in review.json).
Reviews with no tokens are skipped by Lucene as empty, so its N can be slightly smaller than ours.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_reviews, tokenize  # noqa: E402
from search_engine.run_queries import QUERIES  # noqa: E402

_JDK = "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
if os.path.isdir(_JDK):  # Homebrew openjdk@21 (macOS); elsewhere set JAVA_HOME and JVM_PATH yourself
    os.environ.setdefault("JAVA_HOME", _JDK)
    os.environ.setdefault("JVM_PATH", os.path.join(_JDK, "lib", "server", "libjvm.dylib"))
os.environ.setdefault("OPENAI_API_KEY", "unused")  # pyserini.search builds an OpenAI client at import; never called


def prepare(a):
    os.makedirs(a.out, exist_ok=True)
    n = 0
    with open(os.path.join(a.out, "docs.jsonl"), "w", encoding="utf-8") as f:
        for d, r in enumerate(load_reviews(a.reviews, a.limit)):
            f.write(json.dumps({"id": str(d), "contents": " ".join(tokenize(r["text"]))}) + "\n")
            n += 1
    print(f"wrote {n} docs -> {a.out}/docs.jsonl")


def index(a):
    subprocess.run([sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
                    "--input", a.docs, "--index", a.out, "--generator", "DefaultLuceneDocumentGenerator",
                    "--threads", str(a.threads), "--pretokenized"], check=True)


def lucene_search(searcher, query, k):
    return [(int(h.docid), h.score) for h in searcher.search(" ".join(tokenize(query)), k)]


def compare(a):
    from pyserini.analysis import JWhiteSpaceAnalyzer
    from pyserini.search.lucene import LuceneSearcher

    from search_engine.index import InvertedIndex
    from search_engine.ranking import bm25

    ours = InvertedIndex.load(a.our_index)
    searcher = LuceneSearcher(a.index)
    searcher.set_analyzer(JWhiteSpaceAnalyzer())
    searcher.set_bm25(a.k1, a.b)  # same k1/b as ours
    for q in a.queries:
        mine, lucene = bm25(ours, q, a.k + 1, a.k1, a.b), lucene_search(searcher, q, a.k + 1)  # one extra to see the cut-off gap
        sm, sl = [d for d, _ in mine[:a.k]], [d for d, _ in lucene[:a.k]]
        print(f"=== {q!r}  overlap@{a.k}={len(set(sm) & set(sl))}/{a.k}  same order={sm == sl}  top1 same={sm[:1] == sl[:1]}")
        gap = lambda r: (r[a.k - 1][1] - r[a.k][1]) / r[a.k - 1][1] if len(r) > a.k else float("nan")
        print(f"   relative score gap between rank {a.k} and {a.k + 1}: ours {gap(mine):.4f} lucene {gap(lucene):.4f}")
        for rank, ((dm, s1), (dl, s2)) in enumerate(zip(mine[:3], lucene[:3]), 1):
            print(f"   {rank}. ours doc {dm} {s1:8.3f} | lucene doc {dl} {s2:8.3f}")
    compare_tfidf(a, ours, searcher)


def compare_tfidf(a, ours, searcher):
    """Lucene ClassicSimilarity (its built-in TF-IDF) vs our cosine TF-IDF. These are different TF-IDF variants
    (see the ClassicSimilarity javadoc for its exact formula), so only the rankings are comparable, and low overlap is expected."""
    from jnius import autoclass

    from search_engine.ranking import tfidf_cosine

    searcher.object.searcher.setSimilarity(autoclass("org.apache.lucene.search.similarities.ClassicSimilarity")())
    print("\n--- TF-IDF: ours (lnc.ltc cosine) vs Lucene ClassicSimilarity")
    for q in a.queries:
        mine, lucene = tfidf_cosine(ours, q, a.k), lucene_search(searcher, q, a.k)
        sm, sl = [d for d, _ in mine], [d for d, _ in lucene]
        print(f"=== {q!r}  overlap@{a.k}={len(set(sm) & set(sl))}/{a.k}  top1 same={sm[:1] == sl[:1]}")
    searcher.set_bm25(a.k1, a.b)  # restore the BM25 scorer: the similarity above was switched on the shared searcher


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("reviews"); p.add_argument("--limit", type=int); p.add_argument("--out", required=True)
    p.set_defaults(fn=prepare)
    p = sub.add_parser("index"); p.add_argument("docs"); p.add_argument("--out", required=True); p.add_argument("--threads", type=int, default=4)
    p.set_defaults(fn=index)
    p = sub.add_parser("compare"); p.add_argument("index"); p.add_argument("our_index")
    p.add_argument("--queries", nargs="*", default=QUERIES); p.add_argument("-k", type=int, default=10)
    p.add_argument("--k1", type=float, default=1.2); p.add_argument("--b", type=float, default=0.75)
    p.set_defaults(fn=compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
