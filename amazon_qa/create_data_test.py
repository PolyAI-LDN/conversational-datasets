"""Tests for create_data.py."""

import json
import shutil
import tempfile
import unittest
from glob import glob
from os import path

import tensorflow as tf

from amazon_qa import create_data

_TEST_DATA = [
    {
        'question': "A A A",
        'answer': "B B B",
        'asin': "3",  # gets put in test set.
    },
    {
        # Duplicate QA should not create duplicate examples.
        'question': "A A A",
        'answer': "B B B",
        'asin': "4",
    },
    {
        'question': "A A A A A A",  # too many words, will be skipped.
        'answer': "B B B",
        'asin': "4",
    },
    {
        'questions': [
            {
                'questionText': "C C C",
                'answers': [
                    {'answerText': "D D D"},
                    {'answerText': "E E E"},
                    {'answerText': "E E"}  # not enough words, will be skipped.
                ]
            },
            {
                'questionText': "F F F",
                'answers': [
                    {'answerText': "G G G"},
                ]
            },
        ],
        'asin': "1",  # gets put in train set.
    },
    {
        'questions': [
            {
                'questionText': "H H H",
                'answers': [
                    {'answerText': "I I I"},
                ]
            },
        ],
        'asin': "2",  # gets put in train set.
    },
]


class CreateDataPipelineTest(unittest.TestCase):

    def setUp(self):
        self._temp_dir = tempfile.mkdtemp()
        self.maxDiff = None

    def tearDown(self):
        shutil.rmtree(self._temp_dir)

    def test_run(self):
        # These filenames are chosen so that their hashes will cause them to
        # be put in the train and test set respectively.
        with open(path.join(self._temp_dir, "input-000"), "w") as f:
            for obj in _TEST_DATA:
                f.write(("%s\n" % obj).encode("utf-8"))

        create_data.run(argv=[
            "--runner=DirectRunner",
            "--file_pattern={}/input*".format(self._temp_dir),
            "--output_dir=" + self._temp_dir,
            "--dataset_format=TF",
            "--num_shards_test=2",
            "--num_shards_train=2",
            "--min_words=3",
            "--max_words=5",
            "--train_split=0.5",
        ])

        self.assertItemsEqual(
            [path.join(self._temp_dir, expected_file) for expected_file in
             ["train-00000-of-00002.tfrecord",
              "train-00001-of-00002.tfrecord"]],
            glob(path.join(self._temp_dir, "train-*"))
        )
        self.assertItemsEqual(
            [path.join(self._temp_dir, expected_file) for expected_file in
             ["test-00000-of-00002.tfrecord",
              "test-00001-of-00002.tfrecord"]],
            glob(path.join(self._temp_dir, "test-*"))
        )

        train_examples = self._read_examples("train-*")
        expected_train_examples = [
            self.create_example("1", "C C C", "D D D"),
            self.create_example("1", "C C C", "E E E"),
            self.create_example("1", "F F F", "G G G"),
            self.create_example("2", "H H H", "I I I"),
        ]
        self.assertItemsEqual(
            expected_train_examples,
            train_examples
        )

        test_examples = self._read_examples("test-*")
        expected_test_examples = [
            self.create_example("3", "A A A", "B B B"),
        ]
        self.assertItemsEqual(
            expected_test_examples,
            test_examples
        )

    def create_example(self, product_id, question, answer):
        example = tf.train.Example()
        example.features.feature['product_id'].bytes_list.value.append(
            product_id.encode("utf-8"))
        example.features.feature['context'].bytes_list.value.append(
            question.encode("utf-8"))
        example.features.feature['response'].bytes_list.value.append(
            answer.encode("utf-8"))
        return example

    def _read_examples(self, pattern):
        examples = []
        for file_name in glob(path.join(self._temp_dir, pattern)):
            for record in tf.io.tf_record_iterator(file_name):
                example = tf.train.Example()
                example.ParseFromString(record)
                examples.append(example)
        return examples

    def test_run_json(self):
        # These filenames are chosen so that their hashes will cause them to
        # be put in the train and test set respectively.
        with open(path.join(self._temp_dir, "input-000"), "w") as f:
            for obj in _TEST_DATA:
                f.write(("%s\n" % obj).encode("utf-8"))

        create_data.run(argv=[
            "--runner=DirectRunner",
            "--file_pattern={}/input*".format(self._temp_dir),
            "--output_dir=" + self._temp_dir,
            "--dataset_format=JSON",
            "--num_shards_test=2",
            "--num_shards_train=2",
            "--min_words=3",
            "--max_words=5",
            "--train_split=0.5",
        ])

        self.assertItemsEqual(
            [path.join(self._temp_dir, expected_file) for expected_file in
             ["train-00000-of-00002.json",
              "train-00001-of-00002.json"]],
            glob(path.join(self._temp_dir, "train-*"))
        )
        self.assertItemsEqual(
            [path.join(self._temp_dir, expected_file) for expected_file in
             ["test-00000-of-00002.json",
              "test-00001-of-00002.json"]],
            glob(path.join(self._temp_dir, "test-*"))
        )

        train_examples = self._read_json_examples("train-*")
        expected_train_examples = [
            self.create_json_example("1", "C C C", "D D D"),
            self.create_json_example("1", "C C C", "E E E"),
            self.create_json_example("1", "F F F", "G G G"),
            self.create_json_example("2", "H H H", "I I I"),
        ]
        self.assertItemsEqual(
            expected_train_examples,
            train_examples
        )

        test_examples = self._read_json_examples("test-*")
        expected_test_examples = [
            self.create_json_example("3", "A A A", "B B B"),
        ]
        self.assertItemsEqual(
            expected_test_examples,
            test_examples
        )

    def create_json_example(self, product_id, question, answer):
        return {
            'product_id': product_id,
            'context': question,
            'response': answer,
        }

    def _read_json_examples(self, pattern):
        examples = []
        for file_name in glob(path.join(self._temp_dir, pattern)):
            for line in open(file_name):
                examples.append(json.loads(line))
        return examples


if __name__ == "__main__":
    unittest.main()
