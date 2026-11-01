"""Download only business.json and review.json of the Yelp Open Dataset from Kaggle.

Setup: pip install kagglehub, and provide Kaggle credentials via KAGGLE_API_TOKEN or
KAGGLE_USERNAME + KAGGLE_KEY, set in the environment or in SourceCode/.env (see .env.example),
or via ~/.kaggle/kaggle.json.
Files land in ../data (outside SourceCode, so they are not part of the submission zip).
"""
import argparse
import os
import shutil

import kagglehub



def load_dotenv(path):
    """Minimal .env reader (KEY=VALUE lines); real environment variables take precedence."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


HANDLE = "yelp-dataset/yelp-dataset"
FILES = {
    "yelp_academic_dataset_business.json": "business.json",
    "yelp_academic_dataset_review.json": "review.json",
}


def main():
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    ap = argparse.ArgumentParser(description="Download Yelp business.json and review.json")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for remote, local in FILES.items():
        dest = os.path.join(a.out, local)
        if os.path.exists(dest) and not a.force:
            print(f"skip {dest} (exists)")
            continue
        # one call per file: only these two are fetched, not the full dataset
        src = kagglehub.dataset_download(HANDLE, path=remote, force_download=a.force)
        shutil.copyfile(src, dest)
        print(f"{remote} -> {dest}")


if __name__ == "__main__":
    main()
