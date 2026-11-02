"""Check deterministic nested sampling on the Yelp review dataset.

Run from ``SourceCode`` for a quick check over 100,000 reviews:
    python3 -m dataset_analysis.sampling_demo

Scan the complete review file:
    python3 -m dataset_analysis.sampling_demo --limit 0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dataset_analysis.sampling import SAMPLE_THRESHOLDS, sample_memberships


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_review_path() -> Path:
    """Return the standard data path, with support for the original filename."""
    standard_path = PROJECT_ROOT / "data" / "review.json"
    original_path = (
        PROJECT_ROOT / "data" / "raw" / "yelp_academic_dataset_review.json"
    )
    return standard_path if standard_path.is_file() else original_path


def count_nested_samples(path: Path, limit: int | None) -> tuple[int, Counter[str]]:
    """Stream reviews and count membership without storing review text."""
    counts: Counter[str] = Counter()
    reviews_read = 0

    with path.open(encoding="utf-8") as review_file:
        for line_number, line in enumerate(review_file, start=1):
            if limit is not None and reviews_read >= limit:
                break

            try:
                review = json.loads(line)
                review_id = review["review_id"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Invalid review on line {line_number}: {error}") from error

            for sample_name, included in sample_memberships(review_id).items():
                if included:
                    counts[sample_name] += 1
            reviews_read += 1

    return reviews_read, counts


def parse_args() -> argparse.Namespace:
    default_path = default_review_path()
    parser = argparse.ArgumentParser(
        description="Verify deterministic nested samples of Yelp reviews."
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=default_path,
        help=f"Review JSONL path (default: {default_path})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="Maximum reviews to scan; use 0 for the complete file (default: 100000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be 0 or a positive integer")
    if not args.reviews.is_file():
        raise SystemExit(f"Review file not found: {args.reviews}")

    limit = None if args.limit == 0 else args.limit
    total, counts = count_nested_samples(args.reviews, limit)

    print(f"Reviews scanned: {total:,}")
    print("Sampling rule: first 8 hex digits of SHA-256(review_id) / 0xFFFFFFFF")
    print()
    print(f"{'Sample':<8} {'Reviews':>12} {'Observed':>12} {'Target':>10}")
    for sample_name, target in SAMPLE_THRESHOLDS.items():
        observed = counts[sample_name] / total if total else 0.0
        print(
            f"{sample_name:<8} {counts[sample_name]:>12,} "
            f"{observed:>11.2%} {target:>9.0%}"
        )


if __name__ == "__main__":
    main()
