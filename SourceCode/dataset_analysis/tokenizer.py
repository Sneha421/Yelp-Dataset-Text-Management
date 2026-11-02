"""Custom regex tokenizer for noisy Yelp review text.

The tokenizer preserves contractions, hyphenated words, emoji sequences,
URLs, email addresses, prices/numbers, and sentiment-bearing !/? runs.

Try it from ``SourceCode`` with:
    python3 -m dataset_analysis.tokenizer "The-best-food wasn't cheap 😊!"
"""

from __future__ import annotations

import argparse
import unicodedata

import regex


TOKEN_PATTERN = regex.compile(
    r"""
    # Web links. Do not absorb common sentence-ending punctuation.
    https?://[^\s<>()]+(?<![.,!?;:])
    |
    # Email addresses.
    [\w.+-]+@[\w.-]+\.[A-Za-z]{2,}
    |
    # Prices, integers, decimals, percentages, and values such as 20kg.
    [$£€¥]?\d+(?:[.,]\d+)*(?:%|[A-Za-z]+)?
    |
    # Flag emoji, which consist of two regional-indicator characters.
    \p{Regional_Indicator}{2}
    |
    # Emoji, including skin-tone modifiers and joined family/group emoji.
    \p{Extended_Pictographic}
    (?:\uFE0F|\p{Emoji_Modifier})?
    (?:
        \u200D
        \p{Extended_Pictographic}
        (?:\uFE0F|\p{Emoji_Modifier})?
    )*
    |
    # Unicode words with optional contractions and hyphenated components.
    [\p{L}\p{M}]+
    (?:'[\p{L}\p{M}]+)*
    (?:-[\p{L}\p{M}]+(?:'[\p{L}\p{M}]+)*)*
    |
    # Keep runs of ! and ? because they can express emphasis/sentiment.
    [!?]+
    """,
    regex.VERBOSE | regex.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize equivalent Unicode characters before tokenization."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = (
        normalized.replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("‐", "-")
        .replace("‑", "-")
        .replace("–", " ")
        .replace("—", " ")
    )
    return normalized


def custom_tokenize(text: str) -> list[str]:
    """Return normalized tokens from one Yelp review.

    Ordinary tokens are case-folded to reduce vocabulary duplication. URLs
    retain their original case because URL paths can be case-sensitive.
    """
    normalized = normalize_text(text)
    tokens = TOKEN_PATTERN.findall(normalized)

    return [
        token
        if token.lower().startswith(("http://", "https://"))
        else token.casefold()
        for token in tokens
    ]


def compare_tokenizers(text: str) -> dict[str, list[str]]:
    """Run the custom, NLTK whitespace, and NLTK WordPunct tokenizers."""
    try:
        from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer
    except ImportError as error:
        raise RuntimeError(
            "NLTK is required for comparison. Install it with: "
            "python3 -m pip install nltk"
        ) from error

    return {
        "Custom": custom_tokenize(text),
        "Whitespace": WhitespaceTokenizer().tokenize(text),
        "WordPunct": WordPunctTokenizer().tokenize(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare tokenizers on a noisy Yelp-style sentence."
    )
    parser.add_argument("text", help="Text to tokenize; wrap it in quotes")
    args = parser.parse_args()

    print(f"Input: {args.text}\n")
    for name, tokens in compare_tokenizers(args.text).items():
        print(f"{name:<10}: {tokens}")


if __name__ == "__main__":
    main()
