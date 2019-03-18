"""Evaluate baseline models on conversational datasets.

For usage see README.md.
"""

import abc
import argparse
import enum
import random

import glog
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import baselines


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
    return parser.parse_args()


class Method(enum.Enum):
    TF_IDF = 1
    BM25 = 2

    def to_method_object(self):
        """Convert the enum to an instance of `BaselineMethod`."""
        if self == self.TF_IDF:
            return baselines.tf_idf.TfIdfMethod(apply_bm25_transform=False)
        elif self == self.BM25:
            return baselines.tf_idf.TfIdfMethod(apply_bm25_transform=True)
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


class BaselineMethod(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, examples):
        """Perform any training steps using the (context, response) examples.

        Args:
            examples: a list of `(context, response)` string tuples, containing
                examples that can be used to set the parameters of this method.
        """
        pass

    @abc.abstractmethod
    def rank_responses(self, contexts, responses):
        """Rank the responses for each context.

        Args:
            contexts: a list of strings giving the contexts to use.
            responses: a list of responses to rank, of the same length
                as `contexts`. These are shuffled, to help avoid cheating.

        Returns:
            predictions: a list of integers, giving the predictions indices
                between 0 and `len(contexts)` that the method predicts for
                assigning responses to the contexts. Explicitly, the method
                predicts that `reponse[predictions[i]]` is the correct response
                for `context[i]`.
        """
        pass


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

    glog.info(
        "Final computed 1-of-%i accuracy is %.1f%%",
        recall_k,
        100 * accuracy_numerator / accuracy_denominator
    )


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

    unique_c, unique_r = [], []
    seen_contexts = set()

    for context, response in zip(contexts, responses):
        if context in seen_contexts:
            continue
        seen_contexts.add(context)
        unique_c.append(context)
        unique_r.append(response)

    return unique_c, unique_r


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
    _evaluate_method(method, args.recall_k, contexts, responses)
