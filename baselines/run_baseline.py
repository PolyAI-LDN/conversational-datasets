"""Evaluate baseline models on conversational datasets.

For usage see README.md.
"""

import argparse
import csv
import enum
import random

import glog
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from baselines import tf_idf, vector_based


def _parse_args():
    """Parse command-line args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=Method.from_string, choices=list(Method), required=True,
        help="The baseline method to use.")
    parser.add_argument(
        "--recall_k", type=int,
        default=100, help="The value of k to compute recall at.")
    parser.add_argument(
        "--train_dataset", type=str, required=True,
        help="File pattern of train set.")
    parser.add_argument(
        "--train_size", type=int, default=10000,
        help="Number of examples from the training set to use in training.")
    parser.add_argument(
        "--test_dataset", type=str, required=True,
        help="File pattern of test set.")
    parser.add_argument(
        "--eval_num_batches", type=int, default=500,
        help="Number of batches to use in the evaluation.")
    parser.add_argument(
        "--output_file", type=str,
        help="Optional file to output result as a CSV row.")
    return parser.parse_args()


class Method(enum.Enum):
    # Keyword based methods.
    TF_IDF = 1
    BM25 = 2

    # Vector similarity based methods.
    USE_SIM = 3
    USE_LARGE_SIM = 4
    ELMO_SIM = 5

    # Vector mapping methods.
    USE_MAP = 6
    USE_LARGE_MAP = 7
    ELMO_MAP = 8

    def to_method_object(self):
        """Convert the enum to an instance of `BaselineMethod`."""
        if self == self.TF_IDF:
            return tf_idf.TfIdfMethod(apply_bm25_transform=False)
        elif self == self.BM25:
            return tf_idf.TfIdfMethod(apply_bm25_transform=True)
        elif self == self.USE_SIM:
            return vector_based.VectorSimilarityMethod(
                encoder=vector_based.TfHubEncoder(
                    "https://tfhub.dev/google/"
                    "universal-sentence-encoder/2"))
        elif self == self.USE_LARGE_SIM:
            return vector_based.VectorSimilarityMethod(
                encoder=vector_based.TfHubEncoder(
                    "https://tfhub.dev/google/"
                    "universal-sentence-encoder-large/3"))
        elif self == self.ELMO_SIM:
            return vector_based.VectorSimilarityMethod(
                encoder=vector_based.TfHubEncoder(
                    "https://tfhub.dev/google/elmo/1"))
        elif self == self.USE_MAP:
            return vector_based.VectorMappingMethod(
                encoder=vector_based.TfHubEncoder(
                    "https://tfhub.dev/google/"
                    "universal-sentence-encoder/2"))
        elif self == self.USE_LARGE_MAP:
            return vector_based.VectorMappingMethod(
                encoder=vector_based.TfHubEncoder(
                    "https://tfhub.dev/google/"
                    "universal-sentence-encoder-large/3"))
        elif self == self.ELMO_MAP:
            return vector_based.VectorMappingMethod(
                encoder=vector_based.TfHubEncoder(
                    "https://tfhub.dev/google/elmo/1"))
        raise ValueError("Unknown method {}".format(self))

    def __str__(self):
        """String representation to use in argparse help text."""
        return self.name

    @staticmethod
    def from_string(s):
        """Convert a string parsed from argparse to an enum instance."""
        try:
            return Method[s]
        except KeyError:
            raise ValueError()


def _evaluate_method(method, recall_k, contexts, responses):
    accuracy_numerator = 0.0
    accuracy_denominator = 0.0
    for i in tqdm(range(0, len(contexts), recall_k)):
        context_batch = contexts[i:i + recall_k]
        responses_batch = responses[i:i + recall_k]
        if len(context_batch) != recall_k:
            break

        # Shuffle the responses.
        permutation = np.arange(recall_k)
        np.random.shuffle(permutation)
        context_batch_shuffled = [context_batch[j] for j in permutation]

        predictions = method.rank_responses(
            context_batch_shuffled, responses_batch)
        if predictions.shape != (recall_k, ):
            raise ValueError(
                "Predictions returned by method should have shape ({}, ), "
                "but saw {}".format(recall_k, predictions.shape))
        accuracy_numerator += np.equal(predictions, permutation).mean()
        accuracy_denominator += 1.0

    accuracy = 100 * accuracy_numerator / accuracy_denominator
    glog.info(
        "Final computed 1-of-%i accuracy is %.1f%%",
        recall_k, accuracy
    )
    return accuracy


def _load_data(file_pattern, num_examples):
    """Load contexts and responses from the given conversational dataset."""
    contexts = []
    responses = []
    complete = False
    with tqdm(total=num_examples) as progress_bar:
        file_names = tf.gfile.Glob(file_pattern)
        random.shuffle(file_names)
        if not file_names:
            raise ValueError(
                "No files matched pattern {}".format(file_pattern))
        for file_name in file_names:
            glog.info("Reading %s", file_name)
            for record in tf.python_io.tf_record_iterator(file_name):
                example = tf.train.Example()
                example.ParseFromString(record)
                contexts.append(
                    example.features.feature[
                        'context'].bytes_list.value[0].decode("utf-8"))
                responses.append(
                    example.features.feature[
                        'response'].bytes_list.value[0].decode("utf-8"))
                progress_bar.update(1)
                if len(contexts) >= num_examples:
                    complete = True
                    break
            if complete:
                break
    glog.info("Read %i examples", len(contexts))
    if not complete:
        glog.warning(
            "%i examples were requested, but dataset only contains %i.",
            num_examples, len(contexts))

    return contexts, responses


if __name__ == "__main__":
    args = _parse_args()
    method = args.method.to_method_object()
    glog.info("Loading training data")
    contexts, responses = _load_data(args.train_dataset, args.train_size)

    glog.info("Training %s method", args.method)
    method.train(contexts, responses)

    glog.info("Loading test data")
    contexts, responses = _load_data(
        args.test_dataset, args.eval_num_batches * args.recall_k)

    glog.info("Running evaluation")
    accuracy = _evaluate_method(method, args.recall_k, contexts, responses)

    if args.output_file is not None:
        with open(args.output_file, "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                args.method, args.train_dataset, args.test_dataset,
                args.recall_k, accuracy
            ])
