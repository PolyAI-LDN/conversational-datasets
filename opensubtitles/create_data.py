"""A Dataflow script for creating sentence pair data from text files.

For usage see README.md.
"""


import argparse
import hashlib
import logging
import os
import re
import uuid
from functools import partial
from os import path

import apache_beam as beam
import tensorflow as tf
from apache_beam import pvalue
from apache_beam.io.filesystems import FileSystems
from apache_beam.io.tfrecordio import WriteToTFRecord
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions


def _parse_args(argv=None):
    """Parse command-line args."""

    def _positive_int(value):
        """Define a positive integer ArgumentParser type."""
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError(
                "Value must be positive, {} was passed.".format(value))
        return value

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentence_files",
        required=True,
        help="The Google cloud storage file pattern of text files containing "
             "one sentence per line.")
    parser.add_argument(
        "--num_extra_contexts",
        default=10,
        help="The maximum number of extra contexts in an example.")
    parser.add_argument(
        "--min_length",
        default=9, type=_positive_int,
        help="The minimum length of a context / response to include.")
    parser.add_argument(
        "--max_length",
        default=127, type=_positive_int,
        help="The maximum length of a context / response to include.")
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory to write the dataset.")
    parser.add_argument(
        "--train_split", default=0.9,
        type=float,
        help="The proportion of data to put in the training set.")
    parser.add_argument(
        "--num_shards_test", default=100,
        type=_positive_int,
        help="The number of shards for the test set.")
    parser.add_argument(
        "--num_shards_train", default=1000,
        type=_positive_int,
        help="The number of shards for the train set.")

    return parser.parse_known_args(argv)


def _should_skip(line, min_length, max_length):
    """Whether a line should be skipped depending on the length."""
    return len(line) < min_length or len(line) > max_length


def create_example(previous_lines, line, file_id):
    """Creates serialized tensorflow examples with multi-line context

    The examples will include:
        file_id: the name of the file where these lines were obtained.
        response: the current line text
        context: the previous line text
        context/0: 2 lines before
        context/1: 3 lines before, etc.
    """
    example = tf.train.Example()

    example.features.feature['file_id'].bytes_list.value.append(
        file_id.encode("utf-8"))

    context = previous_lines[-1]
    example.features.feature['context'].bytes_list.value.append(
        context.encode("utf-8"))

    extra_contexts = previous_lines[:-1]
    for i, context in enumerate(extra_contexts[::-1]):
        example.features.feature[
            'context/{}'.format(i)].bytes_list.value.append(
            context.encode("utf-8"))

    example.features.feature['response'].bytes_list.value.append(
        line.encode("utf-8"))

    return example


def _preprocess_line(line):
    line = line.decode("utf-8")

    # Remove the first word if it is followed by colon (speaker names)
    # NOTE: this wont work if the speaker's name has more than one word
    line = re.sub('(?:^|(?:[.!?]\\s))(\\w+):', "", line)

    # Remove anything between brackets (corresponds to acoustic events).
    line = re.sub("[\\[(](.*?)[\\])]", "", line)

    # Strip blanks hyphens and line breaks
    line = line.strip(" -\n")

    return line


def _create_examples_from_file(file_name, min_length, max_length,
                               num_extra_contexts):
    _, file_id = path.split(file_name)
    previous_lines = []
    for line in FileSystems.open(file_name, "application/octet-stream"):
        line = _preprocess_line(line)
        if not line:
            continue

        should_skip = _should_skip(
            line,
            min_length=min_length,
            max_length=max_length)

        if previous_lines:
            should_skip |= _should_skip(
                previous_lines[-1],
                min_length=min_length,
                max_length=max_length)

            if not should_skip:
                example = create_example(previous_lines, line, file_id)
                yield example.SerializeToString()

        previous_lines.append(line)
        if len(previous_lines) > num_extra_contexts + 1:
            del previous_lines[0]


def _shuffle_examples(examples):
    examples |= ("add random key" >> beam.Map(
        lambda example: (uuid.uuid4(), example)))
    examples |= ("group by key" >> beam.GroupByKey())
    examples |= ("get shuffled values" >> beam.FlatMap(lambda t: t[1]))
    return examples


class _TrainTestSplitFn(beam.DoFn):
    """Splits an input PCollection of serialized examples into train and test.

    This uses the file id (name) to compute the split, so that examples from
    the same file are in the same set. The split is deterministic based on
    the file id, so that multiple runs produce the same result.
    """

    TRAIN_TAG = "train"
    TEST_TAG = "test"

    def __init__(self, train_split=0.9, num_buckets=4096):
        super(_TrainTestSplitFn, self).__init__()
        self._train_split = train_split
        self._num_buckets = num_buckets

    def process(self, serialized_example):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        file_id, = example.features.feature['file_id'].bytes_list.value
        split_value = self._split_value(file_id)
        split = (
            self.TRAIN_TAG if split_value < self._train_split else
            self.TEST_TAG)
        yield pvalue.TaggedOutput(split, serialized_example)

    def _split_value(self, file_id):
        """Compute a value from 0 to 1 used to compute the split."""
        md5 = hashlib.md5()
        md5.update(file_id)
        md5_digest = int(md5.hexdigest(), 16)
        return (
            (1 + md5_digest % self._num_buckets)
            / float(self._num_buckets)
        )


def run(argv=None):
    """Run the beam pipeline."""
    args, pipeline_args = _parse_args(argv)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    p = beam.Pipeline(options=pipeline_options)

    sentence_files_match = FileSystems.match([args.sentence_files])[0]
    sentence_files = [
        file_metadata.path
        for file_metadata in sentence_files_match.metadata_list]
    logging.info("Reading %i files from %s.",
                 len(sentence_files), args.sentence_files)
    assert len(sentence_files) > 0
    sentence_files = p | beam.Create(sentence_files)
    serialized_examples = sentence_files | "create examples" >> beam.FlatMap(
        partial(_create_examples_from_file,
                min_length=args.min_length,
                max_length=args.max_length,
                num_extra_contexts=args.num_extra_contexts)
    )

    serialized_examples = _shuffle_examples(serialized_examples)

    serialized_examples |= "split train and test" >> beam.ParDo(
        _TrainTestSplitFn(args.train_split)).with_outputs(
            _TrainTestSplitFn.TEST_TAG, _TrainTestSplitFn.TRAIN_TAG)

    (serialized_examples[_TrainTestSplitFn.TRAIN_TAG]
     | "write train" >> WriteToTFRecord(
         os.path.join(args.output_dir, "train"),
         file_name_suffix=".tfrecords",
         num_shards=args.num_shards_train))
    (serialized_examples[_TrainTestSplitFn.TEST_TAG]
     | "write test" >> WriteToTFRecord(
         os.path.join(args.output_dir, "test"),
         file_name_suffix=".tfrecords",
         num_shards=args.num_shards_test))

    result = p.run()
    result.wait_until_finish()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
