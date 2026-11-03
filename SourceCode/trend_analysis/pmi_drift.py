"""Yearly document-level PMI drift for a target Yelp aspect term."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset_analysis.tokenizer import custom_tokenize


STOPWORDS = set(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself just me more most mustn't my myself
    no nor not of off on once only or other ought our ours ourselves out over
    own same shan't she she'd she'll she's should shouldn't so some such than
    that that's the their theirs them themselves then there there's these they
    they'd they'll they're they've this those through to too under until up
    very was wasn't we we'd we'll we're we've were weren't what what's when
    when's where where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself yourselves
    """.split()
)


def extract_year(date_value: object) -> int | None:
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


def candidate_terms(text: str) -> set[str]:
    """Return unique content-word candidates for document-frequency PMI."""
    return {
        token.casefold()
        for token in custom_tokenize(text)
        if len(token) >= 3 and token.isalpha() and token.casefold() not in STOPWORDS
    }


def collect_pmi_counts(
    review_path: Path,
    target: str = "service",
    limit: int | None = None,
    progress_every: int = 100_000,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[
    Counter[int],
    Counter[int],
    dict[int, Counter[str]],
    dict[int, Counter[str]],
    dict[str, int],
]:
    """Collect yearly document frequencies in one streaming pass."""
    normalized_target = target.casefold()
    if not normalized_target.isalpha():
        raise ValueError("Target must be one alphabetic token")
    yearly_reviews: Counter[int] = Counter()
    yearly_target_reviews: Counter[int] = Counter()
    yearly_document_frequency: dict[int, Counter[str]] = defaultdict(Counter)
    yearly_cooccurrence: dict[int, Counter[str]] = defaultdict(Counter)
    metadata = {"reviews_scanned": 0, "reviews_used": 0, "invalid_rows": 0}

    for _, review in _iter_reviews(review_path, limit):
        metadata["reviews_scanned"] += 1
        if progress_every and metadata["reviews_scanned"] % progress_every == 0:
            print(
                f"[PMI] scanned {metadata['reviews_scanned']:,} reviews; "
                f"used {metadata['reviews_used']:,}",
                file=sys.stderr,
            )
        year = extract_year(review.get("date"))
        text = review.get("text")
        if year is None or not isinstance(text, str):
            metadata["invalid_rows"] += 1
            continue
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue

        terms = candidate_terms(text)
        yearly_reviews[year] += 1
        yearly_document_frequency[year].update(terms)
        if normalized_target in terms:
            yearly_target_reviews[year] += 1
            yearly_cooccurrence[year].update(terms - {normalized_target})
        metadata["reviews_used"] += 1

    if metadata["reviews_scanned"] == 0 or metadata["reviews_used"] == 0:
        raise ValueError("No usable reviews were processed")
    return (
        yearly_reviews,
        yearly_target_reviews,
        dict(yearly_document_frequency),
        dict(yearly_cooccurrence),
        metadata,
    )


def calculate_pmi_rows(
    yearly_reviews: Counter[int],
    yearly_target_reviews: Counter[int],
    yearly_document_frequency: dict[int, Counter[str]],
    yearly_cooccurrence: dict[int, Counter[str]],
    target: str = "service",
    top_n: int = 10,
    minimum_document_frequency: int = 50,
    minimum_cooccurrence: int = 10,
    minimum_target_reviews: int = 50,
) -> tuple[list[dict[str, int | float | str]], list[dict[str, int]]]:
    """Calculate and rank yearly PMI associations."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    rows: list[dict[str, int | float | str]] = []
    summaries: list[dict[str, int]] = []
    normalized_target = target.casefold()

    for year in sorted(yearly_reviews):
        total = yearly_reviews[year]
        target_count = yearly_target_reviews[year]
        candidates: list[tuple[float, int, str, int]] = []
        if target_count >= minimum_target_reviews:
            for term, joint_count in yearly_cooccurrence.get(year, {}).items():
                term_count = yearly_document_frequency[year][term]
                if (
                    term_count < minimum_document_frequency
                    or joint_count < minimum_cooccurrence
                ):
                    continue
                pmi = math.log2(
                    (joint_count * total) / (target_count * term_count)
                )
                candidates.append((pmi, joint_count, term, term_count))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected = candidates[:top_n]
        for rank, (pmi, joint_count, term, term_count) in enumerate(
            selected, start=1
        ):
            rows.append(
                {
                    "year": year,
                    "rank": rank,
                    "target": normalized_target,
                    "term": term,
                    "pmi": pmi,
                    "total_reviews": total,
                    "target_reviews": target_count,
                    "term_reviews": term_count,
                    "cooccurring_reviews": joint_count,
                }
            )
        summaries.append(
            {
                "year": year,
                "total_reviews": total,
                "target_reviews": target_count,
                "eligible_candidates": len(candidates),
                "reported_candidates": len(selected),
            }
        )
    validate_pmi_rows(rows, summaries, top_n)
    return rows, summaries


def validate_pmi_rows(
    rows: list[dict[str, int | float | str]],
    summaries: list[dict[str, int]],
    top_n: int,
) -> None:
    """Reject impossible ranks, counts, or non-finite PMI values."""
    by_year: dict[int, list[dict[str, int | float | str]]] = defaultdict(list)
    for row in rows:
        year = int(row["year"])
        by_year[year].append(row)
        total = int(row["total_reviews"])
        target = int(row["target_reviews"])
        term = int(row["term_reviews"])
        joint = int(row["cooccurring_reviews"])
        if not (0 < joint <= target <= total and joint <= term <= total):
            raise ValueError("PMI counts violate document-frequency constraints")
        if not math.isfinite(float(row["pmi"])):
            raise ValueError("PMI must be finite")
    for year, year_rows in by_year.items():
        ranks = [int(row["rank"]) for row in year_rows]
        if ranks != list(range(1, len(ranks) + 1)) or len(ranks) > top_n:
            raise ValueError(f"Invalid ranking for year {year}")
        scores = [float(row["pmi"]) for row in year_rows]
        if scores != sorted(scores, reverse=True):
            raise ValueError(f"PMI scores are not descending for year {year}")
    if not summaries:
        raise ValueError("No yearly summaries were produced")


def select_plot_years(available_years: list[int], maximum: int = 6) -> list[int]:
    """Choose evenly spaced representative years for a readable heatmap."""
    if len(available_years) <= maximum:
        return available_years
    indices = np.linspace(0, len(available_years) - 1, maximum).round().astype(int)
    return [available_years[index] for index in sorted(set(indices))]


def write_pmi_results(
    output_dir: Path,
    rows: list[dict[str, int | float | str]],
    summaries: list[dict[str, int]],
    plot_years: list[int] | None = None,
) -> tuple[Path, Path, Path]:
    """Write yearly PMI tables and a representative-year heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pmi_by_year.csv"
    summary_path = output_dir / "pmi_year_summary.csv"
    plot_path = output_dir / "pmi_heatmap.png"

    row_fields = [
        "year",
        "rank",
        "target",
        "term",
        "pmi",
        "total_reviews",
        "target_reviews",
        "term_reviews",
        "cooccurring_reviews",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=row_fields)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["pmi"] = f"{float(row['pmi']):.8f}"
            writer.writerow(formatted)
    with summary_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    available = sorted({int(row["year"]) for row in rows})
    selected_years = (
        [year for year in plot_years if year in available]
        if plot_years
        else select_plot_years(available)
    )
    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    if not selected_years:
        axis.text(
            0.5,
            0.5,
            "No PMI terms passed the configured thresholds",
            ha="center",
            va="center",
        )
        axis.axis("off")
    else:
        selected_rows = [
            row for row in rows if int(row["year"]) in selected_years
        ]
        terms = sorted(
            {str(row["term"]) for row in selected_rows},
            key=lambda term: -max(
                float(row["pmi"])
                for row in selected_rows
                if row["term"] == term
            ),
        )
        matrix = np.full((len(terms), len(selected_years)), np.nan)
        term_index = {term: index for index, term in enumerate(terms)}
        year_index = {year: index for index, year in enumerate(selected_years)}
        for row in selected_rows:
            matrix[term_index[str(row["term"])], year_index[int(row["year"])]] = float(
                row["pmi"]
            )
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(masked, aspect="auto", cmap="viridis")
        image.cmap.set_bad(color="#eeeeee")
        axis.set_xticks(range(len(selected_years)), selected_years)
        axis.set_yticks(range(len(terms)), terms)
        axis.set_xlabel("Year")
        axis.set_ylabel("Associated term")
        target = str(rows[0]["target"])
        axis.set_title(f"Yearly PMI Associations with '{target}'")
        figure.colorbar(image, ax=axis, label="PMI")
        if len(terms) <= 30:
            for row_index in range(len(terms)):
                for column_index in range(len(selected_years)):
                    value = matrix[row_index, column_index]
                    if not np.isnan(value):
                        axis.text(
                            column_index,
                            row_index,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if value > np.nanmedian(matrix) else "black",
                        )
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return csv_path, summary_path, plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract yearly PMI associations for a Yelp aspect term."
    )
    parser.add_argument("reviews", type=Path, help="Path to review.json")
    parser.add_argument("--target", default="service")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../results/trend_analysis"),
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all reviews")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-df", type=int, default=50)
    parser.add_argument("--min-cooccurrence", type=int, default=10)
    parser.add_argument("--min-target-reviews", type=int, default=50)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--plot-years", nargs="+", type=int)
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    numeric = (
        args.limit,
        args.top_n,
        args.min_df,
        args.min_cooccurrence,
        args.min_target_reviews,
        args.progress_every,
    )
    if args.limit < 0 or args.progress_every < 0 or any(value < 1 for value in numeric[1:5]):
        raise SystemExit("Limits and thresholds must be positive (limit may be 0)")
    if (
        args.start_year is not None
        and args.end_year is not None
        and args.start_year > args.end_year
    ):
        raise SystemExit("--start-year cannot exceed --end-year")
    if not args.reviews.is_file():
        raise SystemExit(f"Review file not found: {args.reviews}")

    counts = collect_pmi_counts(
        args.reviews,
        args.target,
        None if args.limit == 0 else args.limit,
        args.progress_every,
        args.start_year,
        args.end_year,
    )
    yearly_reviews, target_reviews, document_frequency, cooccurrence, metadata = counts
    rows, summaries = calculate_pmi_rows(
        yearly_reviews,
        target_reviews,
        document_frequency,
        cooccurrence,
        args.target,
        args.top_n,
        args.min_df,
        args.min_cooccurrence,
        args.min_target_reviews,
    )
    paths = write_pmi_results(args.output, rows, summaries, args.plot_years)
    print("Review processing:")
    for key, value in metadata.items():
        print(f"  {key}: {value:,}")
    print(f"Years analyzed: {len(summaries)}")
    print(f"Year-term associations written: {len(rows)}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
