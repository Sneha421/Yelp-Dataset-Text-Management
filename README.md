# Yelp Dataset Text Management (AI6122 Group Assignment)

Text processing, search and mining on the Yelp Open Dataset (`business.json`, `review.json`).

```
SourceCode/                 # only our own code; this folder is zipped for submission
  common.py                 # JSON loaders + tokenize() shared by all tasks
  download_data.py          # fetch the two Yelp files from Kaggle into ./data
  .env.example              # Kaggle credentials template
  dataset_analysis/         # 3.2  Zipf, Heaps, custom tokenizer      (TODO)
    tokenizer.py            # custom Unicode-aware Yelp tokenizer
    sampling.py             # deterministic nested dataset sampling
  search_engine/            # 3.3  inverted index, TF-IDF, BM25
  trend_analysis/           # 3.4  VADER sentiment, PMI co-occurrence (TODO)
  application/              # 3.5  comparison-sentence detector
tests/                      # project-level tokenizer and sampling unit tests
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

## Custom tokenizer

The custom regex tokenizer is designed for noisy Yelp review text. It lowercases ordinary words while preserving contractions
(`wasn't`), hyphenated expressions (`the-best-food`), emoji sequences, URLs, email addresses, prices (`$25.50`), percentages,
and runs of `!` and `?`. It uses Unicode-aware matching through the `regex` package.

Install the project dependencies from the repository root:

```bash
python3 -m pip install -r SourceCode/requirements.txt
```

Import the tokenizer in another Python file. Run project modules from `SourceCode/` so the package imports resolve consistently:

```python
from dataset_analysis.tokenizer import custom_tokenize

text = "The-best-food wasn't cheap 😊!"
tokens = custom_tokenize(text)
print(tokens)
# ['the-best-food', "wasn't", 'cheap', '😊', '!']
```

Compare it interactively with NLTK's Whitespace and WordPunct tokenizers:

```bash
cd SourceCode
python3 -m dataset_analysis.tokenizer 'Amazing service!!! Dinner was $25.50 😍'
```

Use single quotes around terminal input containing `$` so the shell does not treat it as a variable. Run the automated tokenizer
tests with:

```bash
python3 -m unittest discover -s tests -p "test_tokenizer.py" -v
```

The shared `common.tokenize()` function delegates to this custom tokenizer, so the search engine and other group modules use the
same tokenization rules automatically. Rebuild any existing search index after tokenizer changes.

## Reproducible nested sampling

Sampling utilities are available from `dataset_analysis.sampling`:

```python
from dataset_analysis.sampling import sample_memberships

membership = sample_memberships(review["review_id"])
if membership["10%"]:
    # This review is also in the 30%, 50%, and 100% samples.
    pass
```

## Run (from `SourceCode/`)
```
python -m search_engine.index ../data/review.json --out ../results/index_full   # full data, about 9 minutes; an index is a directory (add --limit N for a sample)
python -m search_engine.run_queries ../results/index_full --out ../results/final
python -m application.comparison ../data/review.json --limit 200000 --show 10 --out ../results/flagged.csv
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
$PY -m search_engine.lucene_baseline prepare ../data/review.json --out ../results/lucene_docs   # add --limit N for a sample
$PY -m search_engine.lucene_baseline index ../results/lucene_docs --out ../results/lucene_index
$PY -m search_engine.index ../data/review.json --out ../results/index_full   # the same data as the Lucene input
$PY -m search_engine.lucene_baseline compare ../results/lucene_index ../results/index_full
```
Off macOS/Homebrew, set `JAVA_HOME` and `JVM_PATH` (path to `libjvm`) before running. Lucene reads the same tokens our
`tokenize()` produces, so differences come from scoring only (apart from 29 of the 6,990,280 reviews, which have no tokens and which Lucene skips as empty). Lucene's BM25 omits the `(k1+1)` numerator factor
(its scores are ours divided by about `k1+1` = 2.2) and stores document length in one lossy byte, so the ratio varies slightly and near-tied documents can swap places.
