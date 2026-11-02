"""Tests for deterministic nested review sampling."""

import unittest

from SourceCode.dataset_analysis.sampling import (
    sample_fraction,
    sample_memberships,
    smallest_sample,
)


class SamplingTests(unittest.TestCase):
    def test_fraction_is_deterministic(self) -> None:
        self.assertEqual(sample_fraction("review-123"), sample_fraction("review-123"))

    def test_fraction_is_in_expected_range(self) -> None:
        fraction = sample_fraction("review-123")
        self.assertGreaterEqual(fraction, 0.0)
        self.assertLessEqual(fraction, 1.0)

    def test_samples_are_nested(self) -> None:
        for number in range(1_000):
            memberships = sample_memberships(f"review-{number}")
            if memberships["10%"]:
                self.assertTrue(memberships["30%"])
            if memberships["30%"]:
                self.assertTrue(memberships["50%"])
            if memberships["50%"]:
                self.assertTrue(memberships["100%"])

    def test_every_review_is_in_full_sample(self) -> None:
        self.assertTrue(sample_memberships("review-123")["100%"])

    def test_smallest_sample_is_valid(self) -> None:
        self.assertIn(smallest_sample("review-123"), {"10%", "30%", "50%", "100%"})

    def test_empty_review_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_fraction("")


if __name__ == "__main__":
    unittest.main()
