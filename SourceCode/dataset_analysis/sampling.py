"""Deterministic, nested sampling for Yelp reviews."""

from __future__ import annotations

import hashlib


SAMPLE_THRESHOLDS = {
    "10%": 0.10,
    "30%": 0.30,
    "50%": 0.50,
    "100%": 1.00,
}


def sample_fraction(review_id: str) -> float:
    """Return a stable pseudo-random value in the range [0, 1].

    The first 8 hexadecimal characters of a SHA-256 digest represent a
    32-bit integer. The same review ID therefore always receives the same
    value, regardless of when or where this function is called.
    """
    if not isinstance(review_id, str) or not review_id:
        raise ValueError("review_id must be a non-empty string")

    hash_prefix = hashlib.sha256(review_id.encode("utf-8")).hexdigest()[:8]
    return int(hash_prefix, 16) / 0xFFFFFFFF


def sample_memberships(review_id: str) -> dict[str, bool]:
    """Return membership in the nested 10%, 30%, 50%, and 100% samples."""
    fraction = sample_fraction(review_id)
    return {
        name: fraction < threshold if name != "100%" else True
        for name, threshold in SAMPLE_THRESHOLDS.items()
    }


def smallest_sample(review_id: str) -> str:
    """Return the smallest nested sample containing the review."""
    fraction = sample_fraction(review_id)
    if fraction < SAMPLE_THRESHOLDS["10%"]:
        return "10%"
    if fraction < SAMPLE_THRESHOLDS["30%"]:
        return "30%"
    if fraction < SAMPLE_THRESHOLDS["50%"]:
        return "50%"
    return "100%"
