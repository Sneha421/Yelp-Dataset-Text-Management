import json

from dataset_analysis.tokenizer import custom_tokenize


def tokenize(text):
    """Tokenize text consistently across dataset analysis and search."""
    return custom_tokenize(text)


def load_reviews(path, limit=None):
    """Stream review.json (one JSON object per line). Yields dicts."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def load_business_names(path):
    names = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            names[b["business_id"]] = b["name"]
    return names


def load_business_info(path):
    """business_id -> dict(name, city, state, stars, categories) from business.json."""
    info = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            info[b["business_id"]] = {
                "name": b["name"], "city": b.get("city"), "state": b.get("state"),
                "stars": b.get("stars"), "categories": b.get("categories") or "",
            }
    return info
