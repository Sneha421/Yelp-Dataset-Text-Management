"""Tests for tokenizer statistics and Zipf/Heaps analysis."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from SourceCode.dataset_analysis.tokenizer_statistics import (
    run_tokenizer_statistics,
    write_tokenizer_results,
)
from SourceCode.dataset_analysis.zipf_heaps import (
    build_nested_results,
    collect_disjoint_counts,
    write_analysis_results,
)


class CorpusStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.reviews = self.root / "review.json"
        texts = [
            "Amazing service!!! Dinner was $25.50 😍",
            "The-best-food wasn't cheap, but it was delicious.",
            "Email Food@Test.COM or visit https://example.com/menu.",
            "Ordinary review with friendly staff and good food.",
        ]
        with self.reviews.open("w", encoding="utf-8") as output:
            for number in range(200):
                record = {
                    "review_id": f"review-{number}",
                    "business_id": f"business-{number % 5}",
                    "text": f"{texts[number % len(texts)]} item{number}",
                    "date": "2020-01-01 12:00:00",
                    "stars": 4.0,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_tokenizer_statistics_are_consistent_and_written(self) -> None:
        rows, examples = run_tokenizer_statistics(
            self.reviews,
            progress_every=0,
            example_limit=10,
        )
        self.assertEqual([row["tokenizer"] for row in rows], [
            "custom",
            "whitespace",
            "wordpunct",
        ])
        self.assertEqual({row["reviews"] for row in rows}, {200})
        for row in rows:
            self.assertGreater(row["total_tokens"], 0)
            self.assertLessEqual(row["vocabulary_size"], row["total_tokens"])
            self.assertLessEqual(row["hapax_terms"], row["vocabulary_size"])
        self.assertTrue(examples)

        output = self.root / "tokenizer_output"
        statistics_path, examples_path = write_tokenizer_results(
            output, rows, examples
        )
        self.assertTrue(statistics_path.is_file())
        self.assertTrue(examples_path.is_file())
        with statistics_path.open(encoding="utf-8") as source:
            self.assertEqual(len(list(csv.DictReader(source))), 3)

    def test_nested_statistics_and_plots_are_written(self) -> None:
        buckets, review_buckets, total_reviews = collect_disjoint_counts(
            self.reviews, progress_every=0
        )
        results = build_nested_results(
            buckets,
            review_buckets,
            total_reviews,
            zipf_min_rank=1,
            zipf_max_rank=100,
            top_n=10,
        )
        sample_rows, zipf_rows, _, _, _, heaps_fit = results
        self.assertEqual([row["sample"] for row in sample_rows], [
            "10%",
            "30%",
            "50%",
            "100%",
        ])
        self.assertEqual(sample_rows[-1]["reviews"], 200)
        self.assertTrue(all(float(row["slope"]) <= 0 for row in zipf_rows))
        self.assertGreater(float(heaps_fit["k"]), 0)

        output = self.root / "law_output"
        paths = write_analysis_results(output, *results)
        self.assertEqual(len(paths), 6)
        for path in paths:
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
