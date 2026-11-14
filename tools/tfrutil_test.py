"""Tests for TFRecord sampling boundaries."""

import os
import shutil
import tempfile
import unittest

import mock
import tensorflow as tf
from click.testing import CliRunner

from tools import tfrutil


class SampleSizeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.source = os.path.join(self.directory, "input.tfrecord")
        self.output = os.path.join(self.directory, "output.tfrecord")
        with tf.python_io.TFRecordWriter(self.source) as writer:
            for record in (b"first", b"second", b"third"):
                writer.write(record)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_zero_size_does_not_read_input(self):
        with mock.patch.object(tf.python_io, "tf_record_iterator") as reader:
            reader.side_effect = AssertionError("Zero sample read input")
            result = CliRunner().invoke(tfrutil._cli, [
                "sample", "0", self.source, self.output])
        self.assertEqual(result.exit_code, 0, repr(result.exception))
        self.assertEqual(
            list(tf.python_io.tf_record_iterator(self.output)), [])

    def test_negative_size_is_rejected_without_creating_output(self):
        result = CliRunner().invoke(tfrutil._cli, [
            "sample", "-1", self.source, self.output])
        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(os.path.exists(self.output))

    def test_positive_size_keeps_requested_count(self):
        result = CliRunner().invoke(tfrutil._cli, [
            "sample", "2", self.source, self.output])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(list(tf.python_io.tf_record_iterator(
            self.output))), 2)


if __name__ == "__main__":
    unittest.main()
