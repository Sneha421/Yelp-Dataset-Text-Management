"""Checks the on-disk index and both rankers against straightforward per-document formulas.
Run from SourceCode/:  python -m unittest search_engine.test_ranking -v
"""
import json
import math
import os
import tempfile
import unittest
from collections import Counter

from common import tokenize
from search_engine.index import build_index
from search_engine.ranking import bm25, tfidf_cosine

TEXTS = [
    "Amazing pizza and friendly staff.", "Slow service, but the pizza was amazing amazing amazing.",
    "Gluten free options are limited.", "The staff was rude and the service slow.",
    "Best pizza in town, great service!", "Cockroach in my food. Disgusting.",
    "Pizza pizza pizza, I love pizza.", "Nothing special.",
]


class RankingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rp, bp = os.path.join(cls.tmp.name, "review.json"), os.path.join(cls.tmp.name, "business.json")
        with open(rp, "w") as f:
            for i, t in enumerate(TEXTS):
                f.write(json.dumps({"review_id": f"r{i}", "business_id": "b0", "stars": 4.0, "text": t}) + "\n")
        with open(bp, "w") as f:
            f.write(json.dumps({"business_id": "b0", "name": "Joe's", "city": "Reno", "state": "NV", "stars": 4.0, "categories": "Pizza"}) + "\n")
        cls.idx = build_index(rp, bp, os.path.join(cls.tmp.name, "idx"))
        cls.docs = [Counter(tokenize(t)) for t in TEXTS]

    def test_stats(self):
        self.assertEqual(self.idx.N, len(TEXTS))
        self.assertAlmostEqual(self.idx.avgdl, sum(sum(c.values()) for c in self.docs) / len(TEXTS))
        self.assertEqual(self.idx.df("pizza"), sum("pizza" in c for c in self.docs))
        self.assertEqual(self.idx.doc(1)["name"], "Joe's")
        self.assertIn("Slow service", self.idx.doc(1)["text"])

    def test_bm25_matches_formula(self):
        N, avgdl = len(TEXTS), self.idx.avgdl
        for q, k1, b in (("amazing pizza", 1.2, 0.75), ("slow service staff", 2.0, 0.3), ("pizza", 0.5, 1.0)):
            want = {}
            for d, c in enumerate(self.docs):
                dl = sum(c.values())
                s = 0.0
                for t in set(tokenize(q)):
                    df = sum(t in x for x in self.docs)
                    if df and c[t]:
                        s += math.log(1 + (N - df + 0.5) / (df + 0.5)) * c[t] * (k1 + 1) / (c[t] + k1 * (1 - b + b * dl / avgdl))
                if s:
                    want[d] = s
            got = dict(bm25(self.idx, q, 100, k1, b))
            self.assertEqual(set(got), set(want))
            for d in want:
                self.assertAlmostEqual(got[d], want[d])

    def test_tfidf_matches_formula(self):
        N = len(TEXTS)
        df = lambda t: sum(t in c for c in self.docs)
        dvec = lambda c: {t: 1 + math.log(n) for t, n in c.items()}
        qvec = lambda c: {t: (1 + math.log(n)) * math.log(N / df(t)) for t, n in c.items()}
        q = "amazing pizza service"
        qv = qvec(Counter(t for t in tokenize(q) if df(t)))
        got = dict(tfidf_cosine(self.idx, q, 100))
        for d, c in enumerate(self.docs):
            dv = dvec(c)
            dot = sum(w * dv.get(t, 0.0) for t, w in qv.items())
            if dot:
                cos = dot / (math.sqrt(sum(w * w for w in qv.values())) * math.sqrt(sum(w * w for w in dv.values())))
                self.assertAlmostEqual(got[d], cos)

    def test_ties_break_to_smaller_doc_id(self):
        import numpy as np
        from search_engine.ranking import _topk
        ids = np.arange(10)
        self.assertEqual([d for d, _ in _topk(ids, np.ones(10), 3)], [0, 1, 2])
        self.assertEqual(_topk(ids, np.ones(10), 0), [])

    def test_limit_and_close(self):
        small = build_index(os.path.join(self.tmp.name, "review.json"), os.path.join(self.tmp.name, "business.json"),
                            os.path.join(self.tmp.name, "idx_small"), limit=3)
        self.assertEqual(small.N, 3)
        self.assertIn("Gluten free", small.doc(2)["text"])
        small.close()
        small.close()  # idempotent

    def test_unknown_terms(self):
        self.assertEqual(bm25(self.idx, "zzzz", 5), [])
        self.assertEqual(tfidf_cosine(self.idx, "zzzz", 5), [])


if __name__ == "__main__":
    unittest.main()
