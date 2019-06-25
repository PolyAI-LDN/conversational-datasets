"""Tests for create_data.py."""

import copy
import json
import shutil
import tempfile
import unittest
from glob import glob
from os import path

import tensorflow as tf

from reddit import create_data


class CreateDataPipelineTest(unittest.TestCase):
    """Test running the pipeline end-to-end."""

    def setUp(self):
        self._temp_dir = tempfile.mkdtemp()
        self.maxDiff = None

    def tearDown(self):
        shutil.rmtree(self._temp_dir)

    def test_run(self):
        with open("reddit/testdata/simple_thread.json") as f:
            comments = json.loads(f.read())

        # Duplicate the thread with a different ID, chosing a link_id that
        # will be put in the test set.
        test_comments = []
        for comment in comments:
            test_comment = copy.copy(comment)
            test_comment['link_id'] = "t3_testthread"
            test_comments.append(test_comment)

        create_data.run(
            argv=[
                "--runner=DirectRunner",
                "--reddit_table=ignored",
                "--output_dir=" + self._temp_dir,
                "--dataset_format=TF",
                "--num_shards_test=2",
                "--num_shards_train=2",
                "--min_length=4",
                "--max_length=5",
                "--train_split=0.5",
            ],
            comments=(comments + test_comments)
        )

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
            self._create_example(
                {
                    'context': "AAAA",
                    'context_author': "author-A",
                    'response': "BBBB",
                    'response_author': "author-B",
                    'subreddit': "subreddit-A",
                    'thread_id': 'thread-A',
                }
            ),
            self._create_example(
                {
                    'context/0': "AAAA",
                    'context': "BBBB",
                    'context_author': "author-B",
                    'response': "CCCC",
                    'response_author': "author-C",
                    'subreddit': "subreddit-A",
                    'thread_id': 'thread-A',
                }
            ),
            self._create_example(
                {
                    'context/0': "AAAA",
                    'context': "BBBB",
                    'context_author': "author-B",
                    'response': "DDDD",
                    'response_author': "author-D",
                    'subreddit': "subreddit-A",
                    'thread_id': 'thread-A',
                }
            ),
            self._create_example(
                {
                    'context/1': "AAAA",
                    'context/0': "BBBB",
                    'context': "DDDD",
                    'context_author': "author-D",
                    'response': "EEEE",
                    'response_author': "author-E",
                    'subreddit': "subreddit-A",
                    'thread_id': 'thread-A',
                }
            ),
        ]
        self.assertItemsEqual(expected_train_examples, train_examples)

        expected_test_examples = []
        for example in expected_train_examples:
            example.features.feature['thread_id'].bytes_list.value[0] = (
                "testthread").encode("utf-8")
            expected_test_examples.append(example)

        test_examples = self._read_examples("test-*")
        self.assertItemsEqual(expected_test_examples, test_examples)

    def test_run_json(self):
        with open("reddit/testdata/simple_thread.json") as f:
            comments = json.loads(f.read())

        # Duplicate the thread with a different ID, chosing a link_id that
        # will be put in the test set.
        test_comments = []
        for comment in comments:
            test_comment = copy.copy(comment)
            test_comment['link_id'] = "t3_testthread"
            test_comments.append(test_comment)

        create_data.run(
            argv=[
                "--runner=DirectRunner",
                "--reddit_table=ignored",
                "--output_dir=" + self._temp_dir,
                "--dataset_format=JSON",
                "--num_shards_test=2",
                "--num_shards_train=2",
                "--min_length=4",
                "--max_length=5",
                "--train_split=0.5",
            ],
            comments=(comments + test_comments)
        )

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
            {
                'context': "AAAA",
                'context_author': "author-A",
                'response': "BBBB",
                'response_author': "author-B",
                'subreddit': "subreddit-A",
                'thread_id': 'thread-A',
            },
            {
                'context/0': "AAAA",
                'context': "BBBB",
                'context_author': "author-B",
                'response': "CCCC",
                'response_author': "author-C",
                'subreddit': "subreddit-A",
                'thread_id': 'thread-A',
            },
            {
                'context/0': "AAAA",
                'context': "BBBB",
                'context_author': "author-B",
                'response': "DDDD",
                'response_author': "author-D",
                'subreddit': "subreddit-A",
                'thread_id': 'thread-A',
            },
            {
                'context/1': "AAAA",
                'context/0': "BBBB",
                'context': "DDDD",
                'context_author': "author-D",
                'response': "EEEE",
                'response_author': "author-E",
                'subreddit': "subreddit-A",
                'thread_id': 'thread-A',
            }
        ]
        self.assertItemsEqual(expected_train_examples, train_examples)

        expected_test_examples = []
        for example in expected_train_examples:
            example['thread_id'] = "testthread"
            expected_test_examples.append(example)

        test_examples = self._read_json_examples("test-*")
        self.assertItemsEqual(expected_test_examples, test_examples)

    def _read_examples(self, pattern):
        examples = []
        for file_name in glob(path.join(self._temp_dir, pattern)):
            for record in tf.io.tf_record_iterator(file_name):
                example = tf.train.Example()
                example.ParseFromString(record)
                examples.append(example)
        return examples

    def _read_json_examples(self, pattern):
        examples = []
        for file_name in glob(path.join(self._temp_dir, pattern)):
            for line in open(file_name):
                examples.append(json.loads(line))
        return examples

    @staticmethod
    def _create_example(features):
        example = tf.train.Example()
        for feature_name, feature_value in features.items():
            example.features.feature[feature_name].bytes_list.value.append(
                feature_value.encode("utf-8"))
        return example


class CreateDataTest(unittest.TestCase):
    """Test individual helper functions."""

    def test_trim(self):
        self.assertEqual(
            "Matthew",
            create_data.trim("Matthew Henderson", 7)
        )

    def test_trim_do_not_split_word(self):
        self.assertEqual(
            "Matthew ",
            create_data.trim("Matthew Henderson", 9)
        )

    def test_trim_string_short(self):
        self.assertEqual(
            "Matthew",
            create_data.trim("Matthew", 9)
        )

    def test_trim_long_word(self):
        self.assertEqual(
            "",
            create_data.trim("Matthew", 2)
        )

    def test_normalise_comment(self):
        comment = create_data.normalise_comment(
            {
                'body': "ABC EFG HIJ KLM NOP",
                'score_hidden': None,
                'archived': None,
                'name': None,
                'author_flair_text': None,
                'downs': None,
                'created_utc': "1520704245",
                'subreddit_id': "t5_AAAAA",
                'link_id': "t3_BBBBB",
                'parent_id': "t1_CCCCC",
                'score': "1",
                'retrieved_on': "1525020075",
                'controversiality': "0",
                'gilded': "0",
                'id': "DDDDD",
                'subreddit': "EEEEE",
                'author': "FFFFF",
                'ups': None,
                'distinguished': None,
                'author_flair_css_class': None,
            },
            max_length=16)
        self.assertEqual(
            comment,
            create_data.Comment(
                body="ABC EFG HIJ KLM ",
                thread_id="BBBBB",
                parent_id="CCCCC",
                id="DDDDD",
                body_is_trimmed=True,
                subreddit="EEEEE",
                author="FFFFF",
            )
        )

    def test_linear_paths(self):
        with open("reddit/testdata/thread.json") as f:
            comments = json.loads(f.read())
        comments = [
            create_data.normalise_comment(comment, max_length=127)
            for comment in comments]
        id_to_comment = {comment.id: comment for comment in comments}
        paths = list(create_data.linear_paths(id_to_comment, parent_depth=100))
        self.assertItemsEqual(
            [["dvedzte", "dvfdfd4"], ["dvedzte", "dveh7r5"],
             ["dve3v95", "dvhjrkc"], ["dve3v95", "dvhjrkc", "dvhktmd"],
             ["dve3v95", "dvhjrkc", "dvhktmd", "dvhn7hh"],
             ["dve3v95", "dvhjrkc", "dvhktmd", "dvhn7hh", "dvhvg4m"]],
            paths
        )

    @staticmethod
    def _create_test_comment(id, parent_id):
        return create_data.Comment(
            body="body",
            thread_id="thread_id",
            parent_id=parent_id,
            id=id,
            body_is_trimmed=True,
            subreddit="subreddit",
            author="author",
        )

    def test_linear_paths_with_self_loop(self):
        id_to_comment = {
            "1": self._create_test_comment(id="1", parent_id="1"),
        }
        paths = list(create_data.linear_paths(id_to_comment, parent_depth=100))
        self.assertEqual([], paths)

    def test_linear_paths_with_loop(self):
        id_to_comment = {
            "1": self._create_test_comment(id="1", parent_id="3"),
            "2": self._create_test_comment(id="2", parent_id="1"),
            "3": self._create_test_comment(id="3", parent_id="2"),
        }
        paths = list(create_data.linear_paths(id_to_comment, parent_depth=100))
        self.assertEqual([], paths)

    def test_linear_paths_with_stranded_threads(self):
        """Check that it picks up threads whose parents are missing."""
        id_to_comment = {
            "1": self._create_test_comment(id="1", parent_id="unseen"),
            "2": self._create_test_comment(id="2", parent_id="1"),


            "3": self._create_test_comment(id="3", parent_id="unseen 2"),
            "4": self._create_test_comment(id="4", parent_id="3"),
        }
        paths = list(create_data.linear_paths(id_to_comment, parent_depth=100))
        self.assertItemsEqual([
            ["1", "2"],
            ["3", "4"],
        ], paths)

    def test_long_thread(self):
        """Check there is no issue with long threads (e.g. recursion limits)"""
        id_to_comment = {
            i: self._create_test_comment(id=i, parent_id=i - 1)
            for i in range(2000)
        }
        paths = list(create_data.linear_paths(id_to_comment, parent_depth=10))
        self.assertItemsEqual([
            range(max(i - 11, 0), i)
            for i in range(2, 2001)
        ], paths)


if __name__ == "__main__":
    unittest.main()
