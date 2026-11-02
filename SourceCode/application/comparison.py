"""Comparison-sentence detector for Yelp reviews (rule-based, CLI).

A sentence is a comparison if the reviewer contrasts this business with other places.
Every rule is declared once in RULES (name, weight, regex); a sentence is flagged if the summed weight of the
rules that fire (after SUPPRESSES removes overlapping weaker rules) reaches THRESHOLD.
Run with --describe to print the rule table straight from the code.
"""
import argparse
import collections
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_reviews  # noqa: E402

THRESHOLD = 2
DEFAULT_LIMIT = 10000

OTHER = r"(?:other|another|any|every|most|all|the rest|elsewhere|anywhere|somewhere)"
QUANT = r"(?:other|another|any|every|most|all|many|some|the rest)"
PLACE = (r"(?:places?|restaurants?|spots?|joints?|locations?|branch(?:es)?|stores?|shops?|cafes?|bars?|pizzerias?|"
         r"hotels?|motels?|resorts?|diners?|bakery|bakeries|pubs?|steakhouses?|eater(?:y|ies)|buffets?|salons?|gyms?|"
         r"supermarkets?|markets?|clinics?|dealerships?)")
COMPARATIVE = (r"(?:better|worse|cheaper|pricier|bigger|smaller|faster|slower|tastier|fresher|cleaner|nicer|friendlier|"
               r"greater|superior|inferior|higher|lower|longer|shorter|hotter|colder|sweeter|spicier|healthier|busier|"
               r"quieter|louder|(?:more|less)\s+\w+)")
# After "than" / "as ... as", these mean the reviewer is comparing with an expectation, a clause, the same place at
# another time, a review site, a relative, a holiday, a country or state, or nothing at all, not with another business ("better than expected", "than the last time").
NOT_A_PLACE = (r"(?!\s+(?:i|we|you|they|he|she|it|expected|anticipated|thought|imagined|hoped|wanted|advertised|described|"
               r"promised|before|usual|ever|what|people|last|the last|yelp|google|tripadvisor|groupon|facebook|"
               r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
               r"mom|mama|mother|dad|grandma|grandmother|grandpa|home|thanksgiving|christmas|easter|halloween|"
               r"texas|california|florida|america|europe|italy|mexico|china|japan|new york|nyc|vegas)\b"
               r"|\s+my\s+(?:head|face|hand|hands|plate|stomach|expectations?)\b)")
# What follows "than", "as ... as", "unlike" or "compared to" when the reviewer names or points at another business:
# a capitalised name ("Joe's", "KFC", but not shouted words), "other/any/most places", "my usual spot", "elsewhere".
# "the pizza", "an hour" or "other dishes" do not qualify.
DET = r"(?:(?:the|my|our|a|an)\s+)?"
OWN = r"(?:(?:my|our)\s+)?"  # "the buffet" / "the hotel" is usually part of the same visit, "my favorite stores" is not
CAPNAME = r"(?-i:[A-Z][a-z][\w'&-]*|[A-Z]{2,5}\b)"  # (?-i:...) turns case-insensitivity off for the capitalised name
TARGET = (rf"(?:(?:at|in|from)\s+)?(?:{DET}{CAPNAME}|{DET}{QUANT}\s+(?:\w+\s+){{0,2}}?{PLACE}\b"
          rf"|{OWN}(?!(?:the|a|an)\s)(?:\w+\s+){{0,2}}?{PLACE}\b|(?:elsewhere|anywhere)\b)")
RULES = [
    ("explicit_than", 1, rf"\b{COMPARATIVE}\s+than\b{NOT_A_PLACE}"),
    ("than_target", 2, rf"\b{COMPARATIVE}\s+than\b{NOT_A_PLACE}\s+{TARGET}"),
    ("compared_to", 3, rf"\b(?:compared (?:to|with)|in comparison (?:to|with)|as opposed to)\b{NOT_A_PLACE}\s+{TARGET}"),
    ("instead_vs", 1, r"\b(?:rather than|instead of|versus|vs\.?)\b"),
    ("not_as_as", 3, rf"\b(?:not|nowhere near|never|isn't|wasn't|aren't|weren't|n't)\s+(?:nearly\s+|quite\s+|half\s+)?as\s+\w+\s+as\b{NOT_A_PLACE}\s+{TARGET}"),
    ("as_good_as", 2, rf"\bas\s+(?:good|great|bad|tasty|fresh|cheap|expensive|authentic)\s+as\b{NOT_A_PLACE}\s+{TARGET}"),
    ("unlike", 2, rf"\bunlike\b{NOT_A_PLACE}\s+{TARGET}"),
    ("other_places", 1, rf"\b{OTHER}\s+(?:\w+\s+){{0,2}}?{PLACE}\b"),
    ("superlative_exp", 1, r"\b(?:best|worst|finest)\b[^.!?]{0,60}\b(?:i(?:'ve| have)|we(?:'ve| have)|ever|so far|in (?:town|the (?:city|area|neighborhood)))\b"),
    ("prefer_place", 2, r"\b(?:prefer\s+(?:(?-i:[A-Z]\w+)|to go (?:to|elsewhere)\b|over\b|the other|other|going)|rather (?:go|eat|order) (?:to|at|from|elsewhere|somewhere)|go (?:here|there) over|puts .{1,30} to shame)"),
    ("beats", 2, rf"\b(?:beats|trumps|outshines|rivals)\b{NOT_A_PLACE}\s+{TARGET}"),
    ("prefer", 1, r"\bprefer\b"),
    ("similar_to", 1, r"\b(?:reminds? me of|similar to|same as|just like|like the one|on par with|comparable to|a step above|a notch below)\b"),
    ("previous_visit", 1, r"\b(?:used to go to|we (?:usually|normally) go to|my (?:usual|regular|old) (?:place|spot)|switched (?:from|to))\b"),
    ("cross_city", 1, r"\b(?:back (?:home|in (?-i:[A-Z]\w+))|(?:than|like|as|to) (?:in|at|from) (?-i:[A-Z]\w+))\b"),
]
# When the key rule fires, the listed rules match the same phrase and must not be counted again.
SUPPRESSES = {"not_as_as": ("as_good_as",), "prefer_place": ("prefer",), "compared_to": ("instead_vs",),
              "than_target": ("explicit_than", "other_places", "cross_city")}
RULES = [(n, w, re.compile(p, re.I)) for n, w, p in RULES]

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])|\n+")
_ABBREV = re.compile(r"\b(Dr|Mr|Mrs|Ms|St|Mt|Jr|Sr|Inc|Co|Ave|vs)\.", re.I)


def sentences(text):
    text = _ABBREV.sub("\\1\x00", text)  # protect "Dr." etc. from the splitter
    return [s.replace("\x00", ".").strip() for s in _SPLIT.split(text) if s.strip()]


def classify(sent):
    """(is_comparison, score, names of the rules that counted)."""
    fired = [(n, w) for n, w, rx in RULES if rx.search(sent)]
    names = {n for n, _ in fired}
    dropped = {v for k, vs in SUPPRESSES.items() if k in names for v in vs}
    fired = [(n, w) for n, w in fired if n not in dropped]
    score = sum(w for _, w in fired)
    return score >= THRESHOLD, score, [n for n, _ in fired]


def describe():
    print(f"threshold: a sentence is a comparison if the summed rule weight >= {THRESHOLD}\n")
    print(f"{'rule':<16}{'weight':>6}  {'suppresses':<12} pattern")
    for n, w, rx in RULES:
        sup = ",".join(SUPPRESSES.get(n, ())) or "-"
        print(f"{n:<16}{w:>6}  {sup:<12} {rx.pattern}")


def main():
    ap = argparse.ArgumentParser(description="Detect comparison sentences in Yelp reviews")
    ap.add_argument("reviews", nargs="?", help="review.json (JSON lines)")
    ap.add_argument("--describe", action="store_true", help="print the rule table and exit")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--out", help="CSV of detected sentences")
    ap.add_argument("--show", type=int, default=10, help="print N example sentences")
    a = ap.parse_args()
    if a.describe:
        return describe()
    if not a.reviews:
        ap.error("reviews is required unless --describe is given")

    n_rev = n_sent = n_hit = n_rev_hit = 0
    by_star = collections.defaultdict(lambda: [0, 0])  # stars -> [reviews, reviews with comparison]
    rule_count = collections.Counter()
    rows = []
    for r in load_reviews(a.reviews, a.limit):
        n_rev += 1
        by_star[r["stars"]][0] += 1
        found = False
        for s in sentences(r["text"]):
            n_sent += 1
            hit, score, rules = classify(s)
            if hit:
                n_hit += 1; found = True
                rule_count.update(rules)
                rows.append((r["review_id"], r["business_id"], r["stars"], score, "|".join(rules), s))
        if found:
            n_rev_hit += 1; by_star[r["stars"]][1] += 1

    print(f"threshold={THRESHOLD} rules={len(RULES)}")
    print(f"reviews={n_rev} sentences={n_sent} comparison_sentences={n_hit} ({100*n_hit/max(1,n_sent):.2f}%)")
    print(f"reviews with >=1 comparison: {n_rev_hit} ({100*n_rev_hit/max(1,n_rev):.1f}%)")
    print("by star rating (% reviews containing a comparison):")
    for st in sorted(by_star):
        t, h = by_star[st]
        print(f"  {st}: {100*h/max(1,t):5.1f}%  ({h}/{t})")
    print("rule frequency:", dict(rule_count.most_common()))
    for row in rows[: a.show]:
        print(f"- [{row[3]}; {row[4]}] {row[5][:200]}")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["review_id", "business_id", "stars", "score", "rules", "sentence"]); w.writerows(rows)


if __name__ == "__main__":
    main()
