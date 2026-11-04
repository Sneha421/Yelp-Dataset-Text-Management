AI6122 Group Assignment - Yelp review data analysis and processing
=================================================================
Submission contents: this Readme.txt and the SourceCode/ folder (our own code only; the dataset and third-party
libraries are not included). All commands below are run from inside SourceCode/.

Sections marked [TEAM] are to be completed by the members who wrote those parts (3.2 Dataset Analysis, 3.4 Trend Analysis).


1. THIRD-PARTY LIBRARIES (download links)
-----------------------------------------
Python 3.11 (https://www.python.org/downloads/)
numpy         https://numpy.org/                       inverted index arrays, memory mapping, scoring arithmetic
regex         https://pypi.org/project/regex/           Unicode-aware custom tokenizer
kagglehub     https://github.com/Kaggle/kagglehub      dataset download
matplotlib    https://matplotlib.org/                  plots                                           [TEAM]
nltk          https://www.nltk.org/                    standard tokenizers for comparison              [TEAM]
vaderSentiment https://github.com/cjhutto/vaderSentiment  sentiment lexicon                           [TEAM]
Optional, only for the Lucene comparison in 3.3:
Apache Lucene https://lucene.apache.org/ (used through Pyserini 1.6.0, https://github.com/castorini/pyserini)
Java 21 JDK   https://adoptium.net/ (macOS: brew install openjdk@21)
Dataset       Yelp Open Dataset https://www.yelp.com/dataset (Kaggle mirror: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)


2. INSTALLATION
---------------
  pip install -r requirements.txt            (numpy, regex, kagglehub, matplotlib, nltk, vaderSentiment)

Dataset: only business.json and review.json are used (about 5.4 GB).
  a) Automatic: create a Kaggle API token, copy .env.example to .env and set KAGGLE_API_TOKEN
     (or KAGGLE_USERNAME and KAGGLE_KEY), then run:   python download_data.py
     The files are written to ../data/business.json and ../data/review.json.
  b) Manual: download the two JSON files from https://www.yelp.com/dataset, rename them to business.json and
     review.json and place them in ../data/ (relative to SourceCode/).
Input format: both files are JSON Lines (one JSON object per line). Used fields: review.json: review_id, business_id,
stars, text; business.json: business_id, name, city, state, stars, categories.

Optional Lucene comparison environment (kept separate because Pyserini installs PyTorch):
  python3 -m venv ../.venv-lucene && ../.venv-lucene/bin/pip install pyserini numpy regex


3. TASK 3.2 DATASET ANALYSIS  [TEAM]
------------------------------------
Custom tokenizer: dataset_analysis/tokenizer.py   (python -m dataset_analysis.tokenizer "The-best-food wasn't cheap!")
(Zipf, Heaps and tokenizer comparison usage and sample output to be added by the authors.)


4. TASK 3.3 SEARCH ENGINE (inverted index, TF-IDF, BM25)
--------------------------------------------------------
Files (search_engine/): index.py (inverted index), ranking.py (TF-IDF and BM25), run_queries.py (analysis queries),
lucene_baseline.py (Lucene comparison).

4.1 Build the index (all 6,990,280 reviews: about 9 minutes, 3.0 GB on disk; memory use grows with the number of reviews, about 1 GB per million, so allow several GB of RAM):
  python -m search_engine.index ../data/review.json --out ../results/index_full
Parameters: --limit N builds an index of only the first N reviews (for quick tests); --out is the index directory;
business.json is read from the same folder as review.json.
Sample output:
  docs=6990280 terms=966439 postings=495219157 avgdl=105.9 -> ../results/index_full
  docs = reviews indexed; terms = distinct tokens; postings = (document, term frequency) pairs stored;
  avgdl = average review length in tokens, computed once over the whole collection and used by BM25.
The index is only valid for the tokenizer it was built with; if the tokenizer changes, loading fails with
"tokenizer changed ... rebuild it" and the index must be rebuilt.

4.2 Run the analysis queries on both rankers:
  python -m search_engine.run_queries ../results/index_full --out ../results/final
Parameters: -k top results printed per ranker (default 5); --queries "q1" "q2" ... custom queries (default: the 7
queries in the file); --out DIR writes summary.csv, top_results.csv and bm25_sweep.csv.
Sample output:
  === 'amazing pizza'  (tfidf 41.2ms, bm25 40.0ms median of 5, Jaccard@10=0.00)
   [TF-IDF]
       0.594 len=  10 [Sal's Pizza & Restaurant, Mantua] Best pizza in the Mantua area, Hawaiian pizza is amazing
   [BM25]
       9.829 len=  27 [Pi Pizzeria - Delmar Loop, Saint Louis] I absolutely love Pi Pizza. The pizza is always amazing ...
  Each line is: score, review length in tokens, [business name, city], start of the review text. The header gives the
  query, the median latency of 5 runs per ranker, and the Jaccard overlap of the two top-10 lists (0 = no common
  review, 1 = identical). Then comes a BM25 parameter sweep: for the longest query, the mean length (tokens) of the top 10
  results for k1 in {0.5, 1.2, 2.0} and b in {0, 0.75, 1}; b controls how strongly long reviews are penalised.
Rankers (search_engine/ranking.py), usable from Python:
  tfidf_cosine(index, query, k=10)             SMART lnc.ltc: document (1+log tf), no idf, cosine; query (1+log tf)*log(N/df), cosine
  bm25(index, query, k=10, k1=1.2, b=0.75)     Okapi BM25 with idf = ln(1 + (N-df+0.5)/(df+0.5)) and the (k1+1) numerator
Both return a list of (document id, score), highest first.

4.3 Lucene comparison (optional; needs Java 21 and Pyserini, see section 2):
  PY=../.venv-lucene/bin/python
  $PY -m search_engine.lucene_baseline prepare ../data/review.json --out ../results/lucene_docs
  $PY -m search_engine.lucene_baseline index ../results/lucene_docs --out ../results/lucene_index
  $PY -m search_engine.lucene_baseline compare ../results/lucene_index ../results/index_full
Lucene is given our own tokens (pre-tokenized text, whitespace analyzer), so differences come from scoring only.
Sample output:
  === 'cockroach'  overlap@10=10/10  same order=True  top1 same=True
     1. ours doc 4740457   15.129 | lucene doc 4740457    6.887
  overlap@10 = how many of the 10 results both systems share; same order = identical ranking; the numbers are the
  top results with our score and Lucene's score for the same query (Lucene's BM25 omits the (k1+1) factor, so its scores
  are ours divided by about 2.2; the ranking is unaffected). A final section compares our TF-IDF with Lucene's
  ClassicSimilarity, which is a different TF-IDF variant, so only the rankings are comparable.
Off macOS/Homebrew set JAVA_HOME and JVM_PATH (path to libjvm) before running.


5. TASK 3.4 TREND ANALYSIS  [TEAM]
----------------------------------
(Lexicon sentiment over time and PMI co-occurrence drift: usage and sample output to be added by the authors.)


6. TASK 3.5 APPLICATION: COMPARISON-SENTENCE DETECTOR
-----------------------------------------------------
Finds sentences in which the reviewer compares the business with another business or place ("Better than Geno's.",
"Not as good as the other location.", "I prefer Wawa."). It is a rule-based command-line tool (application/comparison.py).
A sentence is flagged when the summed weight of the rules that match reaches 2.

6.1 Run it:
  python -m application.comparison ../data/review.json --limit 200000 --show 10 --out ../results/flagged.csv
Parameters: --limit N reviews to read (default 10000; use 6990280 for the whole file);
--show N example sentences to print (default 10); --out FILE writes every flagged sentence as CSV;
--describe prints the rule table (no data file needed).
Sample output (first 200,000 reviews):
  reviews=200000 sentences=1504208 comparison_sentences=4832 (0.32%)
  reviews with >=1 comparison: 4654 (2.3%)
  by star rating (% reviews containing a comparison):
    3.0:   3.3%  (737/22648)
  rule frequency: {'than_target': 1872, 'other_places': 1429, ...}
  - [2; prefer_place] Definitely prefer Topa.
  The first two lines are the totals; the star table is the share of reviews of that rating containing at least one
  flagged sentence; rule frequency counts how often each rule contributed. Each example line is
  [score; rules that fired] sentence. On the whole dataset (6,990,280 reviews): 154,105 sentences (0.29%), 147,961 reviews (2.1%).
