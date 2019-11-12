"""Tests for vector_based.py."""

import os
import tempfile
import unittest

import mock
import numpy as np
import tensorflow as tf
from mock import patch

from baselines import vector_based


class TfHubEncoderTest(unittest.TestCase):
    @patch("tensorflow_hub.Module")
    def test_encode_context(self, mock_module_cls):
        mock_module_cls.return_value = lambda x: tf.ones(
            [tf.shape(x)[0], 3])
        encoder = vector_based.TfHubEncoder("test_uri")
        mock_module_cls.assert_called_with("test_uri")

        encodings = encoder.encode_context(["hello"])
        np.testing.assert_allclose([[1, 1, 1]], encodings)

        encodings = encoder.encode_context(["hello", "hi"])
        np.testing.assert_allclose([[1, 1, 1], [1, 1, 1]], encodings)

    @patch("tensorflow_hub.Module")
    def test_encode_response(self, mock_module_cls):
        mock_module_cls.return_value = lambda x: tf.ones(
            [tf.shape(x)[0], 3])
        encoder = vector_based.TfHubEncoder("test_uri")
        mock_module_cls.assert_called_with("test_uri")

        encodings = encoder.encode_response(["hello"])
        np.testing.assert_allclose([[1, 1, 1]], encodings)

        encodings = encoder.encode_response(["hello", "hi"])
        np.testing.assert_allclose([[1, 1, 1], [1, 1, 1]], encodings)


class USEDualEncoderTest(unittest.TestCase):
    """Test USEDualEncoder."""

    @patch("tensorflow_hub.Module")
    def test_encode_context(self, mock_module_cls):

        def mock_fn(inputs, signature, as_dict):
            self.assertTrue(as_dict)
            self.assertIn(signature, {"question_encoder", "response_encoder"})
            if signature == "question_encoder":
                self.assertEqual(["input"], list(inputs.keys()))
                return {'outputs': tf.ones([tf.shape(inputs['input'])[0], 3])}
            else:
                self.assertEqual({"input", "context"}, set(inputs.keys()))
                return {'outputs': None}

        mock_module_cls.return_value = mock_fn

        encoder = vector_based.USEDualEncoder("test_uri")
        mock_module_cls.assert_called_with("test_uri")

        encodings = encoder.encode_context(["hello"])
        np.testing.assert_allclose([[1, 1, 1]], encodings)

    @patch("tensorflow_hub.Module")
    def test_encode_response(self, mock_module_cls):
        def mock_fn(inputs, signature, as_dict):
            self.assertTrue(as_dict)
            self.assertIn(signature, {"question_encoder", "response_encoder"})
            if signature == "response_encoder":
                self.assertEqual({"input", "context"}, set(inputs.keys()))
                return {'outputs': tf.ones([tf.shape(inputs['input'])[0], 3])}
            else:
                self.assertEqual(["input"], list(inputs.keys()))
                return {'outputs': None}

        mock_module_cls.return_value = mock_fn

        encoder = vector_based.USEDualEncoder("test_uri")
        mock_module_cls.assert_called_with("test_uri")

        encodings = encoder.encode_response(["hello"])
        np.testing.assert_allclose([[1, 1, 1]], encodings)


class ConveRTEncoderTest(unittest.TestCase):
    """Test ConveRTEncoder."""

    @patch("tensorflow_hub.Module")
    def test_encode_context(self, mock_module_cls):

        def mock_fn(input, signature=None):
            self.assertIn(signature, {"encode_context", "encode_response"})
            self.assertIsInstance(input, tf.Tensor)
            self.assertEqual(input.dtype, tf.string)
            if signature == "encode_context":
                return tf.ones([tf.shape(input)[0], 3])

        mock_module_cls.return_value = mock_fn

        encoder = vector_based.ConveRTEncoder("test_uri")
        mock_module_cls.assert_called_with("test_uri")

        encodings = encoder.encode_context(["hello"])
        np.testing.assert_allclose([[1, 1, 1]], encodings)

    @patch("tensorflow_hub.Module")
    def test_encode_response(self, mock_module_cls):
        def mock_fn(input, signature=None):
            self.assertIn(signature, {"encode_context", "encode_response"})
            self.assertIsInstance(input, tf.Tensor)
            self.assertEqual(input.dtype, tf.string)
            if signature == "encode_response":
                return tf.ones([tf.shape(input)[0], 3])

        mock_module_cls.return_value = mock_fn

        encoder = vector_based.ConveRTEncoder("test_uri")
        mock_module_cls.assert_called_with("test_uri")

        encodings = encoder.encode_response(["hello"])
        np.testing.assert_allclose([[1, 1, 1]], encodings)


class BERTEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a dummy vocabulary file."""
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "hello", "hi",
        ]
        with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
        cls.vocab_file = vocab_writer.name

    @classmethod
    def tearDownClass(cls):
        """Delete the dummy vocabulary file."""
        os.unlink(cls.vocab_file)

    @patch("tensorflow_hub.Module")
    def test_encode_context(self, mock_module_cls):

        def mock_module(inputs=None, signature=None, as_dict=None):
            self.assertTrue(as_dict)
            if signature == "tokens":
                self.assertEqual(
                    {'input_mask', 'input_ids', 'segment_ids'},
                    inputs.viewkeys())
                batch_size = tf.shape(inputs['input_ids'])[0]
                seq_len = tf.shape(inputs['input_ids'])[1]
                return {
                    'sequence_output': tf.ones([batch_size, seq_len, 3])
                }
            self.assertEqual("tokenization_info", signature)
            return {
                'do_lower_case': tf.constant(True),
                'vocab_file': tf.constant(self.vocab_file),
            }

        mock_module_cls.return_value = mock_module

        encoder = vector_based.BERTEncoder("test_uri")
        self.assertEqual(
            [(("test_uri",), {'trainable': False})] * 2,
            mock_module_cls.call_args_list)

        # Final encodings will just be the count of the tokens in each
        # sentence, repeated 3 times.
        encodings = encoder.encode_context(["hello"])
        np.testing.assert_allclose([[3, 3, 3]], encodings)

        encodings = encoder.encode_context(["hello", "hello hi"])
        np.testing.assert_allclose([[3, 3, 3], [4, 4, 4]], encodings)


class VectorSimilarityMethodTest(unittest.TestCase):
    def test_train(self):
        vector_based.VectorSimilarityMethod(None).train(["x", "y"], ["a", "b"])

    def test_rank_responses(self):
        mock_encoder = mock.Mock()
        mock_encoder.encode_context.return_value = np.asarray([
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ], dtype=np.float32)
        mock_encoder.encode_response.return_value = np.asarray([
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ], dtype=np.float32)

        method = vector_based.VectorSimilarityMethod(mock_encoder)
        assignments = method.rank_responses(
            ["x", "y", "z"],
            ["a", "b", "c"]
        )
        np.testing.assert_allclose([0, 1, 2], assignments)
        mock_encoder.encode_context.assert_has_calls([
            mock.call(["x", "y", "z"]),
        ])
        mock_encoder.encode_response.assert_has_calls([
            mock.call(["a", "b", "c"]),
        ])


class VectorMappingMethodTest(unittest.TestCase):
    def test_train_then_rank(self):
        mock_encoder = mock.Mock()

        def _random_encode(texts):
            return np.random.normal(size=(len(texts), 3))

        mock_encoder.encode_context.side_effect = _random_encode
        mock_encoder.encode_response.side_effect = _random_encode

        method = vector_based.VectorMappingMethod(
            mock_encoder, learning_rates=[1], regularizers=[0])

        # Use 104 elements, so that the encoding must be batched.
        method.train(["context"] * 104, ["response"] * 104)
        mock_encoder.encode_context.assert_has_calls([
            mock.call(["context"] * 100),
            mock.call(["context"] * 4),
        ])
        mock_encoder.encode_response.assert_has_calls([
            mock.call(["response"] * 100),
            mock.call(["response"] * 4),
        ])
        assignments = method.rank_responses(
            ["x", "y", "z"],
            ["a", "b", "c"]
        )
        self.assertEqual((3, ), assignments.shape)
        for id_ in assignments:
            self.assertGreaterEqual(id_, 0)
            self.assertLess(id_, 3)


if __name__ == "__main__":
    unittest.main()
