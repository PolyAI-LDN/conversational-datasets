"""Tests for degenerate response encodings."""

import unittest

import mock
import numpy as np

from baselines import vector_based


class ZeroResponseVectorTest(unittest.TestCase):
    def test_zero_response_does_not_hide_a_matching_response(self):
        encoder = mock.Mock()
        encoder.encode_context.return_value = np.array([[1., 0.]])
        encoder.encode_response.return_value = np.array([[0., 0.], [2., 0.]])
        model = vector_based.VectorSimilarityMethod(encoder)
        with np.errstate(divide="raise", invalid="raise"):
            predictions = model.rank_responses(["context"], ["empty", "match"])
        self.assertEqual(list(predictions), [1])

    def test_encoder_response_arrays_are_not_modified(self):
        encoder = mock.Mock()
        encoder.encode_context.return_value = np.array([[1., 0.]])
        responses = np.array([[3., 4.]])
        encoder.encode_response.return_value = responses
        model = vector_based.VectorSimilarityMethod(encoder)
        model.rank_responses(["context"], ["response"])
        np.testing.assert_array_equal(responses, [[3., 4.]])


if __name__ == "__main__":
    unittest.main()
