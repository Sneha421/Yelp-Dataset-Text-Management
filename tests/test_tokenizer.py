"""Unit tests for the custom Yelp tokenizer."""

import unittest

from SourceCode.dataset_analysis.tokenizer import custom_tokenize, normalize_text


class CustomTokenizerTests(unittest.TestCase):
    def test_contractions_and_curly_apostrophes(self) -> None:
        self.assertEqual(
            custom_tokenize("I don’t think it wasn't good"),
            ["i", "don't", "think", "it", "wasn't", "good"],
        )

    def test_hyphenated_words_stay_together(self) -> None:
        self.assertEqual(
            custom_tokenize("The-best-food was over-priced"),
            ["the-best-food", "was", "over-priced"],
        )

    def test_emoji_and_joined_emoji_are_preserved(self) -> None:
        self.assertEqual(
            custom_tokenize("Great 😊 Family 👨‍👩‍👧"),
            ["great", "😊", "family", "👨‍👩‍👧"],
        )

    def test_price_percentage_and_decimal_are_preserved(self) -> None:
        self.assertEqual(
            custom_tokenize("It cost $12.50 with 10% tax"),
            ["it", "cost", "$12.50", "with", "10%", "tax"],
        )

    def test_url_does_not_absorb_sentence_punctuation(self) -> None:
        self.assertEqual(
            custom_tokenize("Visit https://Example.com/Menu!"),
            ["visit", "https://Example.com/Menu", "!"],
        )

    def test_email_is_preserved_and_case_folded(self) -> None:
        self.assertEqual(
            custom_tokenize("Email Food@Test.COM"),
            ["email", "food@test.com"],
        )

    def test_em_dash_separates_words(self) -> None:
        self.assertEqual(custom_tokenize("food—amazing"), ["food", "amazing"])

    def test_question_and_exclamation_runs_are_preserved(self) -> None:
        self.assertEqual(
            custom_tokenize("Really?! Amazing!!!"),
            ["really", "?!", "amazing", "!!!"],
        )

    def test_empty_text_returns_no_tokens(self) -> None:
        self.assertEqual(custom_tokenize(""), [])

    def test_non_string_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            normalize_text(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
