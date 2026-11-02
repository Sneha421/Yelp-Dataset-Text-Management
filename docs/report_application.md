# 3.5 Application: Comparison-Sentence Detector

Figures are from the full dataset (6,990,280 reviews).

## Task definition

In a *comparison sentence* the reviewer compares the business being reviewed (its food, service or prices) with another business or place that is named or clearly pointed at, for example "Much better than Geno's." or "Not as good as the other location." We treat as *not* comparisons: comparison with an expectation ("better than expected"), with the same place on another visit ("better than last time"), between two items at the same place ("the pasta was better than the pizza"), and generic superlatives that name nobody ("the best sushi I've had"). Our definition is slightly broader than the assignment's "other businesses that they have visited": a reviewer who writes "I prefer Wawa" is counted even though the text does not say they visited it.

The tool is a command-line program. Input: `review.json`. Output: counts of flagged sentences and reviews, the share of reviews with a comparison by star rating, rule frequencies, example sentences, and optionally every flagged sentence as CSV.

## Method

A review is split into sentences (on `.`, `!`, `?` followed by a capital letter, and on line breaks; abbreviations such as "Dr." and "St." are protected). Each sentence is matched against 15 hand-written regular-expression rules, each with a weight. A sentence is flagged when the summed weight of the rules that fire reaches a threshold of 2. Three design choices restrict the rules to comparisons with other businesses:

1. **Target requirement.** Cues such as "better than", "not as good as", "unlike", "compared to" and "beats" only count when what follows looks like another business: a capitalised name ("Joe's", "KFC"), "other/any/most/many/some ... places", or a place noun ("my favorite stores", "most hotels"). "better than the pasta", "longer than an hour", "not as bad as Yelp says" and "unlike the other reviewers" therefore do not fire. Bare names that are relatives, holidays, countries or states, days of the week or review sites ("Mom's cooking", "Thanksgiving dinner", "Texas", "Monday", "Yelp") are excluded, and all-caps words are not treated as names.
2. **Weights.** Strong cues (for example `compared_to`, `not_as_as`, a than-comparison with a target) can flag a sentence alone; weak cues (for example `other_places`, `similar_to`, `cross_city`) have weight 1 and need a second cue.
3. **Suppression.** When a strong rule fires, a table removes the weaker rules that match the same phrase, so overlapping rules on the same phrase are suppressed in the main cases (for example `than_target` removes `explicit_than`, `other_places` and `cross_city`). Not every overlap is covered: "compared to other sushi bars" scores 4, because `compared_to` does not remove `other_places`; this does not change whether the sentence is flagged.

| Rule | Weight | Example it catches |
|---|---|---|
| `than_target` | 2 | "much better than Geno's", "bigger than at other places" |
| `compared_to` | 3 | "compared to other sushi bars" |
| `not_as_as` | 3 | "not as good as Sabrina's" |
| `as_good_as` | 2 | "just as good as Joe's" |
| `unlike` | 2 | "unlike most restaurants" |
| `prefer_place` | 2 | "I'd rather go to a different BBQ spot", "I prefer Wawa" |
| `beats` | 2 | "nothing beats Whole Foods" |
| `other_places`, `superlative_exp`, `similar_to`, `instead_vs`, `previous_visit`, `cross_city`, `prefer`, `explicit_than` | 1 | cues that need a second cue to reach the threshold: "all other places", "the best I've had in town", "reminds me of", "instead of", "used to go to", "back home", "prefer", "better than" with no target |

Core of the detector (`application/comparison.py`):

```python
def classify(sent):
    """(is_comparison, score, names of the rules that counted)."""
    fired = [(n, w) for n, w, rx in RULES if rx.search(sent)]
    names = {n for n, _ in fired}
    dropped = {v for k, vs in SUPPRESSES.items() if k in names for v in vs}
    fired = [(n, w) for n, w in fired if n not in dropped]
    score = sum(w for _, w in fired)
    return score >= THRESHOLD, score, [n for n, _ in fired]

# a rule is declared once: (name, weight, regex); TARGET is the "looks like another business" pattern
("than_target", 2, rf"\b{COMPARATIVE}\s+than\b{NOT_A_PLACE}\s+{TARGET}"),
("compared_to", 3, rf"\b(?:compared (?:to|with)|in comparison (?:to|with)|as opposed to)\b{NOT_A_PLACE}\s+{TARGET}"),
```

`python -m application.comparison --describe` prints the full rule table from the code.

## Usage

```
python -m application.comparison ../data/review.json --limit 200000 --show 10 --out flagged.csv
python -m application.comparison --describe      # print the rule table straight from the code
```

`--limit N` reads the first N reviews (the default is 10,000; a very large value reads the whole file), `--show N` prints N example sentences, and `--out` saves every flagged sentence with its review id, business id, stars, score, rules and text. The output is the totals, the share of reviews with a comparison by star rating, how often each rule fired, and the examples.

## Results on the full dataset

| | |
|---|---|
| Reviews / sentences | 6,990,280 / 53,884,833 |
| Sentences flagged | 154,105 (0.29%) |
| Reviews with at least one comparison | 147,961 (2.1%) |
| Run time | 25 min 33 s (single process) |

Share of reviews containing a comparison, by star rating:

| 1 star | 2 stars | 3 stars | 4 stars | 5 stars |
|---|---|---|---|---|
| 1.2% | 2.4% | 3.4% | 2.9% | 1.8% |

Reviewers compare most in mid-range reviews (2 to 4 stars), and least in the extremes. The most frequent rules are `than_target` (59,160 sentences), `other_places` (45,830) and `compared_to` (21,833).

## Example output

Each flagged sentence is printed with its score and the rules that fired. Exact text from the data:

| Sentence | Rules | Correct? |
|---|---|---|
| "Turtle soup has a strange taste so I definitely prefer Felix's." | `prefer_place` | yes |
| "I had the cheese steak and it was much better than Geno's." | `than_target` | yes |
| "The prices were average compared to other sushi bars, so that won't really help them out either when it comes to bringing me back." | `compared_to`, `other_places` | yes |
| "... the coffe is not as good as Sabrina's." | `not_as_as` | yes |
| "When I walked in, its more bar than restaurant, no big deal, but there were tables." | `than_target` | no: "restaurant" matches the place-noun list, but it compares two parts of the same venue |
| "We are big fans of our smoker back home and cook our steaks this way as well, it's rare to find steak cooked over woodsmoke in most places and this was worth the side trip!" | `other_places`, `cross_city` | no: two weak cues ("most places", "back home") add up to the threshold on unrelated phrases |

## Limitations

- Hand-written English rules; sarcasm and negation beyond "not as ... as" are not handled.
- A capitalised word after "than" is assumed to be a business, so "bigger than Disney" would be flagged; the stoplist covers only relatives, holidays, countries and states, days and review sites.
- Comparisons with an unnamed target ("better than the one in Tucson", "better than the pizza at Joe's") are missed by design.
- Two weak cues can add up to the threshold on unrelated phrases ("it's rare to find steak … in most places … back home").
- This report does not include a quantitative accuracy evaluation. Recall in particular is unknown: comparisons the rules miss are not counted, so the 0.29% flagged rate may understate the share of comparison sentences. Precision could be estimated by labelling a random sample of flagged sentences.
