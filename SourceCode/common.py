import json
import re

# Placeholder tokenizer: swap in the group's custom tokenizer (Section 3.2.2) via tokenize().
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


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
