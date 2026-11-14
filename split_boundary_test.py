"""Tests for fully training or fully test dataset splits."""

import unittest

from amazon_qa import create_data as amazon
from opensubtitles import create_data as subtitles
from reddit import create_data as reddit


class TrainSplitBoundaryTest(unittest.TestCase):
    def test_full_training_split_includes_highest_hash_bucket(self):
        for module, field in ((amazon, "product_id"),
                              (subtitles, "file_id"),
                              (reddit, "thread_id")):
            splitter = module._TrainTestSplitFn(train_split=1)
            example = {field: "7105"}
            self.assertEqual(splitter._split_value("7105"), 1)
            output = list(splitter.process(example))
            self.assertEqual(len(output), 1)
            self.assertEqual(output[0].tag, splitter.TRAIN_TAG)
            self.assertEqual(output[0].value, example)

    def test_zero_training_split_remains_entirely_test(self):
        for module, field in ((amazon, "product_id"),
                              (subtitles, "file_id"),
                              (reddit, "thread_id")):
            splitter = module._TrainTestSplitFn(train_split=0)
            output = list(splitter.process({field: "7105"}))
            self.assertEqual(output[0].tag, splitter.TEST_TAG)


if __name__ == "__main__":
    unittest.main()
