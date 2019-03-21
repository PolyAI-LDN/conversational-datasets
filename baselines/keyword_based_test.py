"""Tests for keyword_based.py."""

import unittest

from baselines import keyword_based


class TfIdfMethodTest(unittest.TestCase):
    def test_train_test(self):
        """Check that it can correctly rank a simple example."""
        method = keyword_based.TfIdfMethod()
        method.train(
            ["hello how are you", "hello how are"],
            ["hello how", "hello"]
        )
        predictions = method.rank_responses(
            ["hello", "how", "are", "you"],
            ["you", "are", "how", "hello"]
        )
        self.assertEqual(
            list(predictions),
            [3, 2, 1, 0]
        )

    def test_train_test_idf(self):
        """Check that the keyword with higher idf counts for more."""
        method = keyword_based.TfIdfMethod()
        method.train(
            ["hello how are you", "hello how are"],
            ["hello how", "hello"]
        )
        predictions = method.rank_responses(
            ["hello you", "hello you"],
            ["hello", "you"]
        )
        self.assertEqual(
            list(predictions),
            [1, 1]  # you is more informative than 'hello'.
        )


class BM25MethodTest(unittest.TestCase):
    def test_train_test_bm25(self):
        """Check that bm25 can correctly rank a simple example."""
        method = keyword_based.BM25Method()
        method.train(
            ["hello how are you", "hello how are"],
            ["hello how", "hello"]
        )
        predictions = method.rank_responses(
            ["hello", "how", "are"],
            ["are", "how", "hello"]
        )
        self.assertEqual(
            list(predictions),
            [2, 1, 0]
        )


if __name__ == "__main__":
    unittest.main()
