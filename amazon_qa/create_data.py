"""A Dataflow script for creating Amazon question/answer data.

For usage see README.md.
"""


import argparse
import ast
import hashlib
import json
import logging
import os
import uuid
from functools import partial

import apache_beam as beam
import tensorflow as tf
from apache_beam import pvalue
from apache_beam.io.textio import ReadFromText, WriteToText
from apache_beam.io.tfrecordio import WriteToTFRecord
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

_TF_FORMAT = "TF"
_JSON_FORMAT = "JSON"


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
        "--file_pattern",
        required=True,
        help="File pattern for amazon qa files on Google cloud storage.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory to write the dataset on Google cloud storage.",
    )
    parser.add_argument(
        "--dataset_format",
        choices={_TF_FORMAT, _JSON_FORMAT},
        default="TF",
        help="The dataset format to write. 'TF' for serialized tensorflow "
             "examples in TFRecords. 'JSON' for text files with one JSON "
             "object per line."
    )
    parser.add_argument(
        "--max_words",
        type=_positive_int,
        default=59,
        help="Maximum number of words a Q or A can have to be included.",
    )
    parser.add_argument(
        "--min_words",
        type=_positive_int,
        default=4,
        help="Minimum number of words a Q or A must have to be included.",
    )
    parser.add_argument(
        "--train_split",
        default=0.9, type=float,
        help="The proportion of data to put in the training set.",
    )
    parser.add_argument(
        "--num_shards_test",
        default=10,
        type=_positive_int,
        help="The number of shards for the test set.",
    )
    parser.add_argument(
        "--num_shards_train",
        default=100,
        type=_positive_int,
        help="The number of shards for the train set.",
    )
    return parser.parse_known_args(argv)


def _create_tuples(qa_object, min_words, max_words):
    """Creates (product_id, question, answer) tuples."""
    if "question" in qa_object:
        question = qa_object['question']
        answer = qa_object['answer']
        product_id = qa_object['asin']
        if (_should_skip(question, min_words, max_words)
                or _should_skip(answer, min_words, max_words)):
            return
        yield (product_id, question, answer)

    elif "questions" in qa_object:
        product_id = qa_object['asin']
        for question_obj in qa_object['questions']:
            question = question_obj['questionText']
            if _should_skip(question, min_words, max_words):
                continue
            for answer_obj in question_obj['answers']:
                answer = answer_obj['answerText']
                if _should_skip(answer, min_words, max_words):
                    continue
                yield (product_id, question, answer)


def _should_skip(text, min_words, max_words):
    # Estimate the number of words by splitting on spaces.
    num_words = len(text.split(" "))
    return num_words < min_words or num_words > max_words


def _create_example(product_id, question, answer):
    """Create an example dictionary."""
    return {
        'product_id': product_id,
        'context': question,
        'response': answer,
    }


def _shuffle_examples(examples):
    examples |= "add random key" >> beam.Map(
        lambda example: (uuid.uuid4(), example)
    )
    examples |= "group by key" >> beam.GroupByKey()
    examples |= "get shuffled values" >> beam.FlatMap(lambda t: t[1])
    return examples


def _features_to_serialized_tf_example(features):
    """Convert a string dict to a serialized TF example.

    The dictionary maps feature names (strings) to feature values (strings).
    """
    example = tf.train.Example()
    for feature_name, feature_value in features.items():
        example.features.feature[feature_name].bytes_list.value.append(
            feature_value.encode("utf-8"))
    return example.SerializeToString()


class _TrainTestSplitFn(beam.DoFn):
    """Splits an input PCollection of examples into train and test.

    This uses the product id to compute the split, so that examples from the
    same product are in the same set. The split is deterministic based on
    prodict id, so that multiple runs produce the same result."""

    TRAIN_TAG = "train"
    TEST_TAG = "test"

    def __init__(self, train_split=0.9, num_buckets=4096):
        super(_TrainTestSplitFn, self).__init__()
        self._train_split = train_split
        self._num_buckets = num_buckets

    def process(self, example):
        split_value = self._split_value(example['product_id'])
        split = (
            self.TRAIN_TAG if split_value < self._train_split else
            self.TEST_TAG)
        yield pvalue.TaggedOutput(split, example)

    def _split_value(self, product_id):
        """Compute a value from 0 to 1 used to compute the split."""
        md5 = hashlib.md5()
        md5.update(product_id)
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

    lines = p | "read qa files" >> ReadFromText(args.file_pattern)

    # The lines are not JSON, but the string representation of python
    # dictionary objects. Parse them with ast.literal_eval.
    json_objects = lines | "parsing dictionaries" >> beam.Map(ast.literal_eval)
    qa_tuples = json_objects | "create tuples" >> beam.FlatMap(
        partial(
            _create_tuples,
            min_words=args.min_words, max_words=args.max_words)
    )

    # Remove duplicate examples.
    qa_tuples |= "key by QA" >> beam.Map(lambda v: (v[1:], v))
    qa_tuples |= "group duplicates" >> beam.GroupByKey()
    qa_tuples |= "remove duplicates" >> beam.Map(lambda v: sorted(v[1])[0])

    # Create the examples.
    examples = qa_tuples | "create examples" >> beam.Map(
        lambda args: _create_example(*args)
    )
    examples = _shuffle_examples(examples)

    examples |= "split train and test" >> beam.ParDo(
        _TrainTestSplitFn(args.train_split)
    ).with_outputs(_TrainTestSplitFn.TEST_TAG, _TrainTestSplitFn.TRAIN_TAG)

    if args.dataset_format == _JSON_FORMAT:
        write_sink = WriteToText
        file_name_suffix = ".json"
        serialize_fn = json.dumps
    else:
        assert args.dataset_format == _TF_FORMAT
        write_sink = WriteToTFRecord
        file_name_suffix = ".tfrecord"
        serialize_fn = _features_to_serialized_tf_example

    for name, tag in [("train", _TrainTestSplitFn.TRAIN_TAG),
                      ("test", _TrainTestSplitFn.TEST_TAG)]:

        serialized_examples = examples[tag] | (
            "serialize {} examples".format(name) >> beam.Map(serialize_fn))
        (
            serialized_examples | ("write " + name)
            >> write_sink(
                os.path.join(args.output_dir, name),
                file_name_suffix=file_name_suffix,
                num_shards=args.num_shards_train,
            )
        )

    result = p.run()
    result.wait_until_finish()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
