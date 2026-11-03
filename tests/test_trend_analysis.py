"""Synthetic tests for VADER sentiment trends and yearly PMI drift."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "SourceCode"))

from trend_analysis.pmi_drift import (  # noqa: E402
    calculate_pmi_rows,
    collect_pmi_counts,
    write_pmi_results,
)
from trend_analysis.sentiment_trends import (  # noqa: E402
    aggregate_sentiment,
    load_sector_businesses,
    write_sentiment_results,
)


class TrendAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.businesses = self.root / "business.json"
        self.reviews = self.root / "review.json"
        businesses = [
            {"business_id": "italian", "categories": "Restaurants, Italian"},
            {"business_id": "fast", "categories": "Restaurants, Fast Food"},
            {"business_id": "both", "categories": "Italian, Fast Food"},
            {"business_id": "other", "categories": "Shopping"},
        ]
        with self.businesses.open("w", encoding="utf-8") as output:
            for business in businesses:
                output.write(json.dumps(business) + "\n")

        review_rows = [
            ("r1", "italian", "2020-01-01", "Excellent friendly service and food", 5),
            ("r2", "italian", "2020-02-01", "Terrible slow service", 1),
            ("r3", "fast", "2020-03-01", "Quick service and good food", 4),
            ("r4", "fast", "2021-01-01", "Bad service and cold food", 1),
            ("r5", "both", "2021-01-01", "Great service", 5),
            ("r6", "other", "2021-01-01", "Friendly staff", 4),
        ]
        # Extra documents make the PMI calculation independently checkable.
        pmi_rows = [
            ("p1", "other", "2022-01-01", "service friendly food", 4),
            ("p2", "other", "2022-01-02", "service friendly", 4),
            ("p3", "other", "2022-01-03", "service slow", 2),
            ("p4", "other", "2022-01-04", "food slow", 2),
        ]
        with self.reviews.open("w", encoding="utf-8") as output:
            for review_id, business_id, date, text, stars in review_rows + pmi_rows:
                output.write(
                    json.dumps(
                        {
                            "review_id": review_id,
                            "business_id": business_id,
                            "date": date,
                            "text": text,
                            "stars": stars,
                        }
                    )
                    + "\n"
                )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_sentiment_join_excludes_sector_overlap(self) -> None:
        mapping, business_summary = load_sector_businesses(self.businesses)
        self.assertEqual(mapping, {"italian": "Italian", "fast": "Fast Food"})
        self.assertEqual(business_summary["overlap_excluded"], 1)
        rows, review_summary = aggregate_sentiment(
            self.reviews, mapping, progress_every=0
        )
        self.assertEqual(review_summary["reviews_matched"], 4)
        self.assertEqual(sum(int(row["review_count"]) for row in rows), 4)
        self.assertTrue(all(-1 <= float(row["mean_compound"]) <= 1 for row in rows))

        output = self.root / "sentiment_output"
        paths = write_sentiment_results(output, rows, minimum_reviews=1)
        self.assertTrue(all(path.is_file() and path.stat().st_size for path in paths))

    def test_pmi_matches_manual_document_frequency_calculation(self) -> None:
        counts = collect_pmi_counts(
            self.reviews,
            target="service",
            progress_every=0,
            start_year=2022,
            end_year=2022,
        )
        rows, summaries = calculate_pmi_rows(
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            target="service",
            top_n=10,
            minimum_document_frequency=1,
            minimum_cooccurrence=1,
            minimum_target_reviews=1,
        )
        friendly = next(row for row in rows if row["term"] == "friendly")
        expected = math.log2((2 * 4) / (3 * 2))
        self.assertAlmostEqual(float(friendly["pmi"]), expected)
        self.assertEqual(friendly["cooccurring_reviews"], 2)
        self.assertEqual(summaries[0]["target_reviews"], 3)

        output = self.root / "pmi_output"
        paths = write_pmi_results(output, rows, summaries, plot_years=[2022])
        self.assertTrue(all(path.is_file() and path.stat().st_size for path in paths))


if __name__ == "__main__":
    unittest.main()
