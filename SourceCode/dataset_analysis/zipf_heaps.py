"""Generate nested-sample Zipf and Heaps analyses for Yelp reviews."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .sampling import SAMPLE_THRESHOLDS, smallest_sample
from .tokenizer import custom_tokenize


SAMPLE_ORDER = ("10%", "30%", "50%", "100%")


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


def collect_disjoint_counts(
    review_path: Path,
    limit: int | None = None,
    progress_every: int = 100_000,
) -> tuple[dict[str, Counter[str]], Counter[str], int]:
    """Count tokens in disjoint hash buckets that later form nested samples."""
    token_buckets = {name: Counter() for name in SAMPLE_ORDER}
    review_buckets: Counter[str] = Counter()
    reviews = 0

    for review_id, text in _iter_reviews(review_path, limit):
        bucket = smallest_sample(review_id)
        token_buckets[bucket].update(custom_tokenize(text))
        review_buckets[bucket] += 1
        reviews += 1
        if progress_every and reviews % progress_every == 0:
            print(f"[zipf/heaps] processed {reviews:,} reviews", file=sys.stderr)

    if reviews == 0:
        raise ValueError("No reviews were processed")
    return token_buckets, review_buckets, reviews


def fit_zipf(
    frequencies: np.ndarray,
    requested_min_rank: int,
    requested_max_rank: int,
) -> dict[str, int | float]:
    """Fit log(frequency) = intercept + slope * log(rank)."""
    vocabulary_size = len(frequencies)
    if vocabulary_size < 2:
        raise ValueError("At least two vocabulary terms are needed for Zipf fitting")
    minimum = min(max(requested_min_rank, 1), vocabulary_size - 1)
    maximum = min(max(requested_max_rank, minimum + 1), vocabulary_size)
    ranks = np.arange(minimum, maximum + 1, dtype=np.float64)
    observed = frequencies[minimum - 1 : maximum].astype(np.float64)
    x = np.log(ranks)
    y = np.log(observed)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total else 1.0
    return {
        "fit_min_rank": minimum,
        "fit_max_rank": maximum,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
    }


def _downsample_curve(
    frequencies: np.ndarray, maximum_points: int = 100_000
) -> tuple[np.ndarray, np.ndarray]:
    size = len(frequencies)
    if size <= maximum_points:
        indices = np.arange(size)
    else:
        indices = np.unique(
            np.geomspace(1, size, num=maximum_points).astype(np.int64) - 1
        )
    return indices + 1, frequencies[indices]


def build_nested_results(
    token_buckets: dict[str, Counter[str]],
    review_buckets: Counter[str],
    total_reviews: int,
    zipf_min_rank: int = 10,
    zipf_max_rank: int = 100_000,
    top_n: int = 100,
) -> tuple[
    list[dict[str, int | float | str]],
    list[dict[str, int | float | str]],
    list[dict[str, int | str]],
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, dict[str, int | float]],
    dict[str, float | int],
]:
    """Combine disjoint buckets and calculate Zipf/Heaps results."""
    cumulative: Counter[str] = Counter()
    cumulative_reviews = 0
    sample_rows: list[dict[str, int | float | str]] = []
    zipf_rows: list[dict[str, int | float | str]] = []
    top_rows: list[dict[str, int | str]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fits: dict[str, dict[str, int | float]] = {}

    for sample_name in SAMPLE_ORDER:
        cumulative.update(token_buckets[sample_name])
        cumulative_reviews += review_buckets[sample_name]
        total_tokens = sum(cumulative.values())
        vocabulary_size = len(cumulative)
        if cumulative_reviews == 0 or total_tokens == 0 or vocabulary_size < 2:
            raise ValueError(
                f"Sample {sample_name} is too small; increase --limit"
            )

        frequencies = np.fromiter(
            cumulative.values(), dtype=np.int64, count=vocabulary_size
        )
        frequencies.sort()
        frequencies = frequencies[::-1]
        fit = fit_zipf(frequencies, zipf_min_rank, zipf_max_rank)
        fits[sample_name] = fit
        curves[sample_name] = _downsample_curve(frequencies)

        sample_row: dict[str, int | float | str] = {
            "sample": sample_name,
            "target_fraction": SAMPLE_THRESHOLDS[sample_name],
            "reviews": cumulative_reviews,
            "review_percentage": 100.0 * cumulative_reviews / total_reviews,
            "total_tokens": total_tokens,
            "vocabulary_size": vocabulary_size,
            "average_tokens_per_review": total_tokens / cumulative_reviews,
            "lexical_diversity": vocabulary_size / total_tokens,
        }
        sample_rows.append(sample_row)
        zipf_rows.append({**sample_row, **fit})
        for rank, (token, frequency) in enumerate(
            cumulative.most_common(top_n), start=1
        ):
            top_rows.append(
                {
                    "sample": sample_name,
                    "rank": rank,
                    "token": token,
                    "frequency": frequency,
                }
            )

    validate_nested_rows(sample_rows, total_reviews)
    heaps_fit = fit_heaps(sample_rows)
    return sample_rows, zipf_rows, top_rows, curves, fits, heaps_fit


def validate_nested_rows(
    rows: list[dict[str, int | float | str]], total_reviews: int
) -> None:
    """Reject impossible or non-nested sample statistics."""
    if [row["sample"] for row in rows] != list(SAMPLE_ORDER):
        raise ValueError("Samples are missing or out of order")
    for field in ("reviews", "total_tokens", "vocabulary_size"):
        values = [int(row[field]) for row in rows]
        if values != sorted(values):
            raise ValueError(f"Nested {field} values must be non-decreasing")
    if int(rows[-1]["reviews"]) != total_reviews:
        raise ValueError("The 100% sample does not contain every review")
    for row in rows:
        if int(row["vocabulary_size"]) > int(row["total_tokens"]):
            raise ValueError("Vocabulary cannot exceed total token count")


def fit_heaps(
    sample_rows: list[dict[str, int | float | str]],
) -> dict[str, float | int]:
    """Fit V = k*N^beta using the four nested sample observations."""
    tokens = np.array(
        [int(row["total_tokens"]) for row in sample_rows], dtype=np.float64
    )
    vocabularies = np.array(
        [int(row["vocabulary_size"]) for row in sample_rows],
        dtype=np.float64,
    )
    if np.any(tokens <= 0) or np.any(vocabularies <= 0):
        raise ValueError("Heaps fitting requires positive N and V values")
    x = np.log(tokens)
    y = np.log(vocabularies)
    beta, intercept = np.polyfit(x, y, 1)
    predicted = intercept + beta * x
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total else 1.0
    return {
        "k": math.exp(float(intercept)),
        "beta": float(beta),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "points": len(sample_rows),
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = {
                key: f"{value:.8f}" if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(formatted)


def write_analysis_results(
    output_dir: Path,
    sample_rows: list[dict[str, int | float | str]],
    zipf_rows: list[dict[str, int | float | str]],
    top_rows: list[dict[str, int | str]],
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    fits: dict[str, dict[str, int | float]],
    heaps_fit: dict[str, float | int],
) -> list[Path]:
    """Write CSV tables and publication-ready PNG figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "sample_statistics.csv"
    zipf_path = output_dir / "zipf_fits.csv"
    heaps_path = output_dir / "heaps_fit.csv"
    top_path = output_dir / "top_tokens.csv"
    zipf_plot_path = output_dir / "zipf_plot.png"
    heaps_plot_path = output_dir / "heaps_plot.png"

    sample_fields = [
        "sample",
        "target_fraction",
        "reviews",
        "review_percentage",
        "total_tokens",
        "vocabulary_size",
        "average_tokens_per_review",
        "lexical_diversity",
    ]
    zipf_fields = sample_fields + [
        "fit_min_rank",
        "fit_max_rank",
        "slope",
        "intercept",
        "r_squared",
    ]
    _write_csv(sample_path, sample_rows, sample_fields)
    _write_csv(zipf_path, zipf_rows, zipf_fields)
    _write_csv(heaps_path, [heaps_fit], list(heaps_fit))
    _write_csv(top_path, top_rows, ["sample", "rank", "token", "frequency"])

    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    for axis, sample_name in zip(axes.flat, SAMPLE_ORDER):
        ranks, frequencies = curves[sample_name]
        fit = fits[sample_name]
        axis.loglog(ranks, frequencies, linewidth=1.1, label="Observed")
        fit_ranks = np.geomspace(
            int(fit["fit_min_rank"]), int(fit["fit_max_rank"]), num=200
        )
        fit_frequencies = np.exp(float(fit["intercept"])) * fit_ranks ** float(
            fit["slope"]
        )
        axis.loglog(fit_ranks, fit_frequencies, "--", label="Log-log fit")
        axis.set_title(
            f"{sample_name} sample: slope={float(fit['slope']):.3f}, "
            f"R²={float(fit['r_squared']):.3f}"
        )
        axis.set_xlabel("Term rank")
        axis.set_ylabel("Term frequency")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.suptitle("Zipf's Law for Nested Yelp Review Samples")
    figure.savefig(zipf_plot_path, dpi=200)
    plt.close(figure)

    token_counts = np.array(
        [int(row["total_tokens"]) for row in sample_rows], dtype=np.float64
    )
    vocabulary_sizes = np.array(
        [int(row["vocabulary_size"]) for row in sample_rows],
        dtype=np.float64,
    )
    fitted_tokens = np.geomspace(token_counts.min(), token_counts.max(), 300)
    fitted_vocabularies = float(heaps_fit["k"]) * fitted_tokens ** float(
        heaps_fit["beta"]
    )
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    axis.loglog(token_counts, vocabulary_sizes, "o", markersize=7, label="Observed")
    axis.loglog(fitted_tokens, fitted_vocabularies, "--", label="Heaps fit")
    for row, token_count, vocabulary_size in zip(
        sample_rows, token_counts, vocabulary_sizes
    ):
        axis.annotate(
            str(row["sample"]),
            (token_count, vocabulary_size),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set_title(
        "Heaps' Law: "
        f"V={float(heaps_fit['k']):.3f}N^{float(heaps_fit['beta']):.3f}, "
        f"R²={float(heaps_fit['r_squared']):.3f}"
    )
    axis.set_xlabel("Tokens, N")
    axis.set_ylabel("Vocabulary size, V")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(heaps_plot_path, dpi=200)
    plt.close(figure)

    return [
        sample_path,
        zipf_path,
        heaps_path,
        top_path,
        zipf_plot_path,
        heaps_plot_path,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Zipf's and Heaps' laws on nested Yelp samples."
    )
    parser.add_argument("reviews", type=Path, help="Path to review.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../results/dataset_analysis"),
        help="Output directory (default: ../results/dataset_analysis)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Maximum reviews; 0 means all"
    )
    parser.add_argument("--zipf-min-rank", type=int, default=10)
    parser.add_argument("--zipf-max-rank", type=int, default=100_000)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0 or args.progress_every < 0:
        raise SystemExit("--limit and --progress-every cannot be negative")
    if args.zipf_min_rank < 1 or args.zipf_max_rank <= args.zipf_min_rank:
        raise SystemExit("Zipf ranks must satisfy 1 <= min < max")
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")
    if not args.reviews.is_file():
        raise SystemExit(f"Review file not found: {args.reviews}")

    limit = None if args.limit == 0 else args.limit
    token_buckets, review_buckets, total_reviews = collect_disjoint_counts(
        args.reviews, limit, args.progress_every
    )
    results = build_nested_results(
        token_buckets,
        review_buckets,
        total_reviews,
        args.zipf_min_rank,
        args.zipf_max_rank,
        args.top_n,
    )
    paths = write_analysis_results(args.output, *results)
    sample_rows, zipf_rows, _, _, _, heaps_fit = results

    print(f"{'Sample':<8} {'Reviews':>12} {'Tokens':>15} {'Vocabulary':>12}")
    for row in sample_rows:
        print(
            f"{str(row['sample']):<8} {int(row['reviews']):>12,} "
            f"{int(row['total_tokens']):>15,} "
            f"{int(row['vocabulary_size']):>12,}"
        )
    print("Zipf fits:")
    for row in zipf_rows:
        print(
            f"  {row['sample']}: slope={float(row['slope']):.4f}, "
            f"R²={float(row['r_squared']):.4f}"
        )
    print(
        f"Heaps fit: k={float(heaps_fit['k']):.4f}, "
        f"beta={float(heaps_fit['beta']):.4f}, "
        f"R²={float(heaps_fit['r_squared']):.4f}"
    )
    print("Outputs:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
