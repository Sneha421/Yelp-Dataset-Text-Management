# Yelp Dataset Text Management (AI6122 Group Assignment)

Text processing, search and mining on the Yelp Open Dataset (`business.json`, `review.json`).

```
SourceCode/                 # only our own code; this folder is zipped for submission
  common.py                 # JSON loaders + tokenize() shared by all tasks
  download_data.py          # fetch the two Yelp files from Kaggle into ./data
  .env.example              # Kaggle credentials template
  dataset_analysis/         # 3.2  Zipf, Heaps, custom tokenizer      (TODO)
  search_engine/            # 3.3  inverted index, TF-IDF, BM25
  trend_analysis/           # 3.4  VADER sentiment, PMI co-occurrence (TODO)
  application/              # 3.5  comparison-sentence detector
data/                       # downloaded dataset (git-ignored)
results/                    # generated figures/tables (git-ignored)
```

## Dataset (not stored in git)
`data/` is git-ignored: each member downloads the two files they need (`business.json`, `review.json`, ~5.4 GB total) from the
[Yelp Open Dataset on Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset). Only these two files are fetched, not the full dataset.

1. Create a Kaggle account and an API token (Kaggle > Settings > API > Create New Token).
2. Install the downloader and set your credentials (the `.env` file is also git-ignored):
   ```
   cd SourceCode
   pip install -r requirements.txt
   cp .env.example .env        # then set KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY)
   ```
   Exporting the same variables in your shell, or having `~/.kaggle/kaggle.json`, also works.
3. Download:
   ```
   python download_data.py     # writes ../data/business.json and ../data/review.json
   ```
   Files that already exist are skipped; use `--force` to re-download or `--out DIR` to choose another location.

Alternatively, download the Yelp JSON files manually from https://www.yelp.com/dataset and rename them to
`data/business.json` and `data/review.json`. Please share one download per group rather than everyone downloading it.

## Run (from `SourceCode/`)
```
python -m search_engine.index ../data/review.json --limit 100000 --out ../results/index_100k   # an index is a directory
python -m search_engine.run_queries ../results/index_100k
python -m unittest search_engine.test_ranking          # ranking checked against per-document formulas
python -m application.comparison ../data/review.json --limit 100000 --annotate 200
python -m application.evaluate annotate.csv
```

Note: an index is only valid for the tokenizer it was built with. If `common.tokenize()` changes, loading an old index fails with
"tokenizer changed ... rebuild it"; rebuild with `python -m search_engine.index`.

## Lucene BM25 baseline (optional, for the 3.3 comparison)
Needs Java 21 and Pyserini, kept in a separate virtualenv because Pyserini pulls in PyTorch.
```
brew install openjdk@21                      # macOS; any JDK 21 works
python3 -m venv .venv-lucene && .venv-lucene/bin/pip install pyserini
cd SourceCode
PY=../.venv-lucene/bin/python
$PY -m search_engine.lucene_baseline prepare ../data/review.json --limit 200000 --out ../results/lucene_docs
$PY -m search_engine.lucene_baseline index ../results/lucene_docs --out ../results/lucene_index
$PY -m search_engine.index ../data/review.json --limit 200000 --out ../results/index_200k
$PY -m search_engine.lucene_baseline compare ../results/lucene_index ../results/index_200k
```
Off macOS/Homebrew, set `JAVA_HOME` and `JVM_PATH` (path to `libjvm`) before running. Lucene reads the same tokens our
`tokenize()` produces, so differences come from scoring only (apart from 2 of the first 200,000 reviews, which have no tokens and which Lucene skips as empty). Lucene's BM25 omits the `(k1+1)` numerator factor
(scores are lower by exactly `k1+1` = 2.2) and stores document length in one lossy byte, so exact order can differ slightly.
