"""Yearly VADER sentiment trends for two Yelp business sectors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


DEFAULT_SECTORS = ("Italian", "Fast Food")


def parse_categories(value: str | None) -> set[str]:
    """Split Yelp's comma-separated categories into normalized labels."""
    if not value:
        return set()
    return {category.strip().casefold() for category in value.split(",") if category.strip()}


def load_sector_businesses(
    business_path: Path,
    sectors: tuple[str, str] = DEFAULT_SECTORS,
) -> tuple[dict[str, str], dict[str, int]]:
    """Map business IDs to exactly one requested sector.

    Businesses matching both sectors are excluded so the two comparison
    groups remain disjoint.
    """
    normalized = {sector.casefold(): sector for sector in sectors}
    mapping: dict[str, str] = {}
    summary = {sector: 0 for sector in sectors}
    summary["overlap_excluded"] = 0
    summary["businesses_scanned"] = 0

    with business_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                business = json.loads(line)
                business_id = business["business_id"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"Invalid business on line {line_number}: {error}"
                ) from error
            summary["businesses_scanned"] += 1
            categories = parse_categories(business.get("categories"))
            matched = [
                original
                for key, original in normalized.items()
                if key in categories
            ]
            if len(matched) == 1:
                mapping[business_id] = matched[0]
                summary[matched[0]] += 1
            elif len(matched) > 1:
                summary["overlap_excluded"] += 1

    if not mapping:
        raise ValueError(
            "No businesses matched the requested sectors; check category names"
        )
    return mapping, summary


def extract_year(date_value: object) -> int | None:
    """Extract a plausible four-digit year from a Yelp date string."""
    if not isinstance(date_value, str) or len(date_value) < 4:
        return None
    prefix = date_value[:4]
    if not prefix.isdigit():
        return None
    year = int(prefix)
    return year if 1900 <= year <= 2100 else None


def _iter_reviews(
    review_path: Path, limit: int | None
) -> Iterable[tuple[int, dict]]:
    with review_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if limit is not None and line_number > limit:
                break
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid review JSON on line {line_number}: {error}"
                ) from error


def aggregate_sentiment(
    review_path: Path,
    business_sectors: dict[str, str],
    limit: int | None = None,
    progress_every: int = 100_000,
    analyzer: SentimentIntensityAnalyzer | None = None,
) -> tuple[list[dict[str, int | float | str]], dict[str, int]]:
    """Score relevant reviews and aggregate VADER sentiment by sector/year."""
    scorer = analyzer or SentimentIntensityAnalyzer()
    aggregates: dict[tuple[str, int], dict[str, float | int]] = defaultdict(
        lambda: {
            "count": 0,
            "sum_compound": 0.0,
            "sum_compound_squared": 0.0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "sum_stars": 0.0,
            "star_count": 0,
        }
    )
    metadata = {
        "reviews_scanned": 0,
        "reviews_matched": 0,
        "invalid_dates": 0,
        "invalid_text": 0,
    }

    for line_number, review in _iter_reviews(review_path, limit):
        metadata["reviews_scanned"] += 1
        if progress_every and metadata["reviews_scanned"] % progress_every == 0:
            print(
                f"[sentiment] scanned {metadata['reviews_scanned']:,} reviews; "
                f"matched {metadata['reviews_matched']:,}",
                file=sys.stderr,
            )
        sector = business_sectors.get(review.get("business_id"))
        if sector is None:
            continue
        year = extract_year(review.get("date"))
        if year is None:
            metadata["invalid_dates"] += 1
            continue
        text = review.get("text")
        if not isinstance(text, str):
            metadata["invalid_text"] += 1
            continue

        compound = float(scorer.polarity_scores(text)["compound"])
        if not -1.0 <= compound <= 1.0:
            raise ValueError(
                f"VADER returned invalid compound score on line {line_number}"
            )
        bucket = aggregates[(sector, year)]
        bucket["count"] += 1
        bucket["sum_compound"] += compound
        bucket["sum_compound_squared"] += compound * compound
        if compound >= 0.05:
            bucket["positive"] += 1
        elif compound <= -0.05:
            bucket["negative"] += 1
        else:
            bucket["neutral"] += 1
        stars = review.get("stars")
        if isinstance(stars, (int, float)):
            bucket["sum_stars"] += float(stars)
            bucket["star_count"] += 1
        metadata["reviews_matched"] += 1

    if metadata["reviews_scanned"] == 0:
        raise ValueError("No reviews were processed")
    if metadata["reviews_matched"] == 0:
        raise ValueError(
            "No reviews matched the selected businesses within the scan limit"
        )

    rows: list[dict[str, int | float | str]] = []
    for (sector, year), values in sorted(
        aggregates.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        count = int(values["count"])
        mean = float(values["sum_compound"]) / count
        if count > 1:
            numerator = float(values["sum_compound_squared"]) - count * mean * mean
            variance = max(numerator / (count - 1), 0.0)
            standard_error = math.sqrt(variance / count)
        else:
            standard_error = 0.0
        confidence_margin = 1.96 * standard_error
        star_count = int(values["star_count"])
        row: dict[str, int | float | str] = {
            "sector": sector,
            "year": year,
            "review_count": count,
            "mean_compound": mean,
            "standard_error": standard_error,
            "ci95_lower": max(-1.0, mean - confidence_margin),
            "ci95_upper": min(1.0, mean + confidence_margin),
            "positive_percentage": 100.0 * int(values["positive"]) / count,
            "neutral_percentage": 100.0 * int(values["neutral"]) / count,
            "negative_percentage": 100.0 * int(values["negative"]) / count,
            "average_stars": (
                float(values["sum_stars"]) / star_count
                if star_count
                else float("nan")
            ),
        }
        rows.append(row)
    validate_sentiment_rows(rows)
    return rows, metadata


def validate_sentiment_rows(rows: list[dict[str, int | float | str]]) -> None:
    """Reject impossible sentiment aggregates."""
    if not rows:
        raise ValueError("No sentiment rows were produced")
    for row in rows:
        count = int(row["review_count"])
        mean = float(row["mean_compound"])
        lower = float(row["ci95_lower"])
        upper = float(row["ci95_upper"])
        percentage_sum = sum(
            float(row[name])
            for name in (
                "positive_percentage",
                "neutral_percentage",
                "negative_percentage",
            )
        )
        if count <= 0 or not -1.0 <= mean <= 1.0:
            raise ValueError("Invalid review count or mean compound score")
        if not -1.0 <= lower <= mean <= upper <= 1.0:
            raise ValueError("Invalid confidence interval")
        if abs(percentage_sum - 100.0) > 1e-6:
            raise ValueError("Sentiment percentages do not sum to 100")


def write_sentiment_results(
    output_dir: Path,
    rows: list[dict[str, int | float | str]],
    minimum_reviews: int = 100,
) -> tuple[Path, Path]:
    """Write yearly aggregates and a two-panel sentiment/count plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sentiment_by_year.csv"
    plot_path = output_dir / "sentiment_by_year.png"
    fields = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.8f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    sectors = sorted({str(row["sector"]) for row in rows})
    figure, (sentiment_axis, count_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )
    plotted = False
    for sector in sectors:
        sector_rows = [
            row
            for row in rows
            if row["sector"] == sector
            and int(row["review_count"]) >= minimum_reviews
        ]
        if not sector_rows:
            sector_rows = [row for row in rows if row["sector"] == sector]
        if not sector_rows:
            continue
        plotted = True
        years = [int(row["year"]) for row in sector_rows]
        means = [float(row["mean_compound"]) for row in sector_rows]
        lowers = [float(row["ci95_lower"]) for row in sector_rows]
        uppers = [float(row["ci95_upper"]) for row in sector_rows]
        counts = [int(row["review_count"]) for row in sector_rows]
        sentiment_axis.plot(years, means, marker="o", label=sector)
        sentiment_axis.fill_between(years, lowers, uppers, alpha=0.18)
        count_axis.plot(years, counts, marker="o", label=sector)

    if not plotted:
        raise ValueError("No sector rows were available for plotting")
    sentiment_axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    sentiment_axis.set_ylabel("Mean VADER compound score")
    sentiment_axis.set_title(
        "Yearly Yelp Review Sentiment by Business Sector "
        f"(minimum {minimum_reviews} reviews shown when available)"
    )
    sentiment_axis.grid(True, alpha=0.25)
    sentiment_axis.legend()
    count_axis.set_xlabel("Year")
    count_axis.set_ylabel("Review count")
    count_axis.set_yscale("log")
    count_axis.grid(True, alpha=0.25)
    count_axis.legend()
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return csv_path, plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot yearly VADER sentiment for two Yelp sectors."
    )
    parser.add_argument("businesses", type=Path, help="Path to business.json")
    parser.add_argument("reviews", type=Path, help="Path to review.json")
    parser.add_argument(
        "--sectors", nargs=2, default=list(DEFAULT_SECTORS), metavar=("A", "B")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../results/trend_analysis"),
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all reviews")
    parser.add_argument("--min-reviews", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0 or args.progress_every < 0 or args.min_reviews < 1:
        raise SystemExit("Limits must be non-negative and --min-reviews positive")
    if not args.businesses.is_file() or not args.reviews.is_file():
        raise SystemExit("Business or review input file was not found")
    sectors = (args.sectors[0], args.sectors[1])
    business_sectors, business_summary = load_sector_businesses(
        args.businesses, sectors
    )
    rows, review_summary = aggregate_sentiment(
        args.reviews,
        business_sectors,
        None if args.limit == 0 else args.limit,
        args.progress_every,
    )
    csv_path, plot_path = write_sentiment_results(
        args.output, rows, args.min_reviews
    )

    print("Business selection:")
    for key, value in business_summary.items():
        print(f"  {key}: {value:,}")
    print("Review processing:")
    for key, value in review_summary.items():
        print(f"  {key}: {value:,}")
    print(f"CSV:  {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
