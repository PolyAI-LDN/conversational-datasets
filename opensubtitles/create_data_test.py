"""Tests for create_data.py."""

import shutil
import tempfile
import unittest
from glob import glob
from os import path

import tensorflow as tf

from opensubtitles import create_data

_TRAIN_FILE = "\n".join([
    "matt: AAAA",  # words followed by colons are stripped.
    "[skip]",  # text in square brackets are removed.
    "BBBB",
    "", "", ""  # empty lines are ignored.
    "CCCC",
    "DDDD [boom!]",
    "123",  # line length will be below the test --min_length.
    "12345",  # line length will be above the test --min_length.
])

_TEST_FILE = """
aaaa
bbbb
cccc
dddd
"""


class CreateDataPipelineTest(unittest.TestCase):

    def setUp(self):
        self._temp_dir = tempfile.mkdtemp()
        self.maxDiff = None

    def tearDown(self):
        shutil.rmtree(self._temp_dir)

    def test_run(self):
        # These filenames are chosen so that their hashes will cause them to
        # be put in the train and test set respectively.
        with open(path.join(self._temp_dir, "input_train.txt"), "w") as f:
            f.write(_TRAIN_FILE.encode("utf-8"))

        with open(path.join(self._temp_dir, "input_test.txt"), "w") as f:
            f.write(_TEST_FILE.encode("utf-8"))

        create_data.run(argv=[
            "--runner=DirectRunner",
            "--sentence_files={}/*.txt".format(self._temp_dir),
            "--output_dir=" + self._temp_dir,
            "--num_shards_test=2",
            "--num_shards_train=2",
            "--min_length=4",
            "--max_length=5",
            "--train_split=0.5",
        ])

        self.assertItemsEqual(
            [path.join(self._temp_dir, expected_file) for expected_file in
             ["train-00000-of-00002.tfrecords",
              "train-00001-of-00002.tfrecords"]],
            glob(path.join(self._temp_dir, "train-*"))
        )
        self.assertItemsEqual(
            [path.join(self._temp_dir, expected_file) for expected_file in
             ["test-00000-of-00002.tfrecords",
              "test-00001-of-00002.tfrecords"]],
            glob(path.join(self._temp_dir, "test-*"))
        )

        train_examples = self._read_examples("train-*")
        expected_train_examples = [
            create_data.create_example(
                ["AAAA"], "BBBB", "input_train.txt"),
            create_data.create_example(
                ["AAAA", "BBBB"], "CCCC", "input_train.txt"),
            create_data.create_example(
                ["AAAA", "BBBB", "CCCC"], "DDDD", "input_train.txt"),
        ]
        self.assertItemsEqual(
            expected_train_examples,
            train_examples
        )

        test_examples = self._read_examples("test-*")
        expected_test_examples = [
            create_data.create_example(
                ["aaaa"], "bbbb", "input_test.txt"),
            create_data.create_example(
                ["aaaa", "bbbb"], "cccc", "input_test.txt"),
            create_data.create_example(
                ["aaaa", "bbbb", "cccc"], "dddd", "input_test.txt"),
        ]
        self.assertItemsEqual(
            expected_test_examples,
            test_examples
        )

    def _read_examples(self, pattern):
        examples = []
        for file_name in glob(path.join(self._temp_dir, pattern)):
            for record in tf.io.tf_record_iterator(file_name):
                example = tf.train.Example()
                example.ParseFromString(record)
                examples.append(example)
        return examples


if __name__ == "__main__":
    unittest.main()
