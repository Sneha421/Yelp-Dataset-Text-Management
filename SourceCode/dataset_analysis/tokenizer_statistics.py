"""Compare the custom, whitespace, and WordPunct tokenizers on Yelp reviews.

The review file is streamed once per tokenizer so only one full vocabulary
counter is held in memory at a time. Results are written outside SourceCode.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import regex
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer

from .tokenizer import custom_tokenize, normalize_text


Tokenizer = Callable[[str], list[str]]
TOKENIZER_NAMES = ("custom", "whitespace", "wordpunct")

NOISE_PATTERNS = {
    "contraction": regex.compile(r"\p{L}+[’']\p{L}+"),
    "hyphenated": regex.compile(r"\b\p{L}+(?:-\p{L}+)+\b"),
    "emoji": regex.compile(
        r"\p{Extended_Pictographic}|\p{Regional_Indicator}{2}"
    ),
    "price": regex.compile(r"[$£€¥]\s*\d"),
    "url": regex.compile(r"https?://|www\.", regex.IGNORECASE),
    "email": regex.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "repeated_punctuation": regex.compile(r"[!?]{2,}"),
}


def _standard_input(text: str) -> str:
    """Apply common normalization/case-folding before standard tokenizers."""
    return normalize_text(text).casefold()


def get_tokenizers() -> dict[str, Tokenizer]:
    """Return fresh tokenizer callables used in the comparison."""
    whitespace = WhitespaceTokenizer()
    wordpunct = WordPunctTokenizer()
    return {
        "custom": custom_tokenize,
        "whitespace": lambda text: whitespace.tokenize(_standard_input(text)),
        "wordpunct": lambda text: wordpunct.tokenize(_standard_input(text)),
    }


def detect_noise_features(text: str) -> list[str]:
    """Return the noisy-text features present in a review excerpt."""
    return [name for name, pattern in NOISE_PATTERNS.items() if pattern.search(text)]


def _iter_reviews(path: Path, limit: int | None) -> Iterable[tuple[str, str]]:
    with path.open(encoding="utf-8") as review_file:
        for line_number, line in enumerate(review_file, start=1):
            if limit is not None and line_number > limit:
                break
            try:
                review = json.loads(line)
                review_id = review["review_id"]
                text = review["text"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"Invalid review on line {line_number}: {error}"
                ) from error
            if not isinstance(review_id, str) or not isinstance(text, str):
                raise ValueError(
                    f"Invalid review_id/text types on line {line_number}"
                )
            yield review_id, text


def _remember_example(
    review_id: str,
    text: str,
    feature_examples: dict[str, tuple[str, str, list[str]]],
    fallback_examples: list[tuple[str, str, list[str]]],
    fallback_limit: int,
) -> None:
    features = detect_noise_features(text)
    if not features:
        return
    item = (review_id, text, features)
    for feature in features:
        feature_examples.setdefault(feature, item)
    if len(fallback_examples) < fallback_limit:
        fallback_examples.append(item)


def _choose_examples(
    feature_examples: dict[str, tuple[str, str, list[str]]],
    fallback_examples: list[tuple[str, str, list[str]]],
    maximum: int,
) -> list[tuple[str, str, list[str]]]:
    chosen: list[tuple[str, str, list[str]]] = []
    seen: set[str] = set()
    ordered = list(feature_examples.values()) + fallback_examples
    for item in ordered:
        review_id = item[0]
        if review_id not in seen:
            chosen.append(item)
            seen.add(review_id)
        if len(chosen) >= maximum:
            break
    return chosen


def analyze_one_tokenizer(
    review_path: Path,
    tokenizer_name: str,
    tokenizer: Tokenizer,
    limit: int | None,
    progress_every: int,
    collect_examples: bool,
    example_limit: int,
) -> tuple[dict[str, int | float | str], list[tuple[str, str, list[str]]]]:
    """Stream the corpus and calculate exact statistics for one tokenizer."""
    frequencies: Counter[str] = Counter()
    total_tokens = 0
    reviews = 0
    feature_examples: dict[str, tuple[str, str, list[str]]] = {}
    fallback_examples: list[tuple[str, str, list[str]]] = []
    started = time.perf_counter()

    for review_id, text in _iter_reviews(review_path, limit):
        tokens = tokenizer(text)
        frequencies.update(tokens)
        total_tokens += len(tokens)
        reviews += 1

        if collect_examples:
            _remember_example(
                review_id,
                text,
                feature_examples,
                fallback_examples,
                max(example_limit * 5, example_limit),
            )
        if progress_every and reviews % progress_every == 0:
            print(
                f"[{tokenizer_name}] processed {reviews:,} reviews",
                file=sys.stderr,
            )

    if reviews == 0:
        raise ValueError("No reviews were processed")

    vocabulary_size = len(frequencies)
    hapax_terms = sum(count == 1 for count in frequencies.values())
    elapsed = time.perf_counter() - started
    row: dict[str, int | float | str] = {
        "tokenizer": tokenizer_name,
        "reviews": reviews,
        "total_tokens": total_tokens,
        "vocabulary_size": vocabulary_size,
        "average_tokens_per_review": total_tokens / reviews,
        "hapax_terms": hapax_terms,
        "hapax_percentage": (
            100.0 * hapax_terms / vocabulary_size if vocabulary_size else 0.0
        ),
        "lexical_diversity": (
            vocabulary_size / total_tokens if total_tokens else 0.0
        ),
        "elapsed_seconds": elapsed,
    }
    validate_statistics_row(row)
    examples = _choose_examples(
        feature_examples, fallback_examples, example_limit
    )
    return row, examples


def validate_statistics_row(row: dict[str, int | float | str]) -> None:
    """Reject internally impossible tokenizer statistics."""
    reviews = int(row["reviews"])
    total_tokens = int(row["total_tokens"])
    vocabulary_size = int(row["vocabulary_size"])
    hapax_terms = int(row["hapax_terms"])
    average = float(row["average_tokens_per_review"])
    if reviews <= 0:
        raise ValueError("reviews must be positive")
    if not 0 <= vocabulary_size <= total_tokens:
        raise ValueError("vocabulary_size must be between 0 and total_tokens")
    if not 0 <= hapax_terms <= vocabulary_size:
        raise ValueError("hapax_terms must be between 0 and vocabulary_size")
    if abs(average - total_tokens / reviews) > 1e-9:
        raise ValueError("average_tokens_per_review is inconsistent")


def run_tokenizer_statistics(
    review_path: Path,
    tokenizer_names: list[str] | None = None,
    limit: int | None = None,
    progress_every: int = 100_000,
    example_limit: int = 20,
) -> tuple[
    list[dict[str, int | float | str]],
    list[tuple[str, str, list[str]]],
]:
    """Analyze requested tokenizers and return statistics plus noisy examples."""
    names = tokenizer_names or list(TOKENIZER_NAMES)
    tokenizers = get_tokenizers()
    unknown = set(names) - set(tokenizers)
    if unknown:
        raise ValueError(f"Unknown tokenizers: {sorted(unknown)}")

    rows: list[dict[str, int | float | str]] = []
    examples: list[tuple[str, str, list[str]]] = []
    for position, name in enumerate(names):
        row, found_examples = analyze_one_tokenizer(
            review_path,
            name,
            tokenizers[name],
            limit,
            progress_every,
            collect_examples=position == 0,
            example_limit=example_limit,
        )
        rows.append(row)
        if position == 0:
            examples = found_examples
        gc.collect()

    review_counts = {int(row["reviews"]) for row in rows}
    if len(review_counts) != 1:
        raise ValueError("Tokenizers did not process identical review counts")
    return rows, examples


def write_tokenizer_results(
    output_dir: Path,
    rows: list[dict[str, int | float | str]],
    examples: list[tuple[str, str, list[str]]],
    example_max_chars: int = 500,
) -> tuple[Path, Path]:
    """Write quantitative statistics and qualitative comparisons as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = output_dir / "tokenizer_statistics.csv"
    example_path = output_dir / "tokenizer_examples.csv"

    fields = [
        "tokenizer",
        "reviews",
        "total_tokens",
        "vocabulary_size",
        "average_tokens_per_review",
        "hapax_terms",
        "hapax_percentage",
        "lexical_diversity",
        "elapsed_seconds",
    ]
    with statistics_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for name in (
                "average_tokens_per_review",
                "hapax_percentage",
                "lexical_diversity",
                "elapsed_seconds",
            ):
                formatted[name] = f"{float(row[name]):.8f}"
            writer.writerow(formatted)

    tokenizers = get_tokenizers()
    example_fields = [
        "example_id",
        "review_id",
        "features",
        "text_excerpt",
        "custom_tokens",
        "whitespace_tokens",
        "wordpunct_tokens",
    ]
    with example_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=example_fields)
        writer.writeheader()
        for number, (review_id, text, features) in enumerate(examples, start=1):
            excerpt = " ".join(text.split())[:example_max_chars]
            writer.writerow(
                {
                    "example_id": number,
                    "review_id": review_id,
                    "features": ";".join(features),
                    "text_excerpt": excerpt,
                    "custom_tokens": json.dumps(
                        tokenizers["custom"](excerpt)[:100], ensure_ascii=False
                    ),
                    "whitespace_tokens": json.dumps(
                        tokenizers["whitespace"](excerpt)[:100], ensure_ascii=False
                    ),
                    "wordpunct_tokens": json.dumps(
                        tokenizers["wordpunct"](excerpt)[:100], ensure_ascii=False
                    ),
                }
            )
    return statistics_path, example_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tokenizers on the Yelp review JSONL corpus."
    )
    parser.add_argument("reviews", type=Path, help="Path to review.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../results/dataset_analysis"),
        help="Output directory (default: ../results/dataset_analysis)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum reviews per tokenizer; 0 means all reviews",
    )
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        choices=TOKENIZER_NAMES,
        default=list(TOKENIZER_NAMES),
    )
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--example-max-chars", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0 or args.examples < 0 or args.progress_every < 0:
        raise SystemExit("--limit, --examples, and --progress-every cannot be negative")
    if args.example_max_chars <= 0:
        raise SystemExit("--example-max-chars must be positive")
    if not args.reviews.is_file():
        raise SystemExit(f"Review file not found: {args.reviews}")

    limit = None if args.limit == 0 else args.limit
    rows, examples = run_tokenizer_statistics(
        args.reviews,
        args.tokenizers,
        limit,
        args.progress_every,
        args.examples,
    )
    statistics_path, example_path = write_tokenizer_results(
        args.output, rows, examples, args.example_max_chars
    )

    print(f"{'Tokenizer':<12} {'Reviews':>12} {'Tokens':>15} {'Vocabulary':>12}")
    for row in rows:
        print(
            f"{str(row['tokenizer']):<12} {int(row['reviews']):>12,} "
            f"{int(row['total_tokens']):>15,} "
            f"{int(row['vocabulary_size']):>12,}"
        )
    print(f"Statistics: {statistics_path}")
    print(f"Examples:   {example_path}")


if __name__ == "__main__":
    main()
