# -*- coding: utf-8 -*-
"""Command line utilities for manipulating tfrecords files.

Usage:

To count the number of examples in a tfrecord file:

    python tfrutil.py size train-00999-of-01000.tfrecords

To sample 10000 examples from a file pattern to an output file:

    python tfrutil.py sample 10000 train-*-of-01000.tfrecords \
        train-sampled.tfrecords

To pretty print the contents of a tfrecord file:

    python tfrutil.py pp train-00999-of-01000.tfrecords

This can accept gs:// file paths, as well as local files.
"""


import codecs
import random
import sys

import click
import six
import tensorflow as tf


@click.group()
def _cli():
    """Command line utilities for manipulating tfrecords files."""
    pass


@_cli.command(name="size")
@click.argument("path", type=str, required=True, nargs=1)
def _size(path):
    """Compute the number of examples in the input tfrecord file."""
    i = 0
    for _ in tf.python_io.tf_record_iterator(path):
        i += 1
    print(i)


@_cli.command(name="sample")
@click.argument("sample_size", type=int, required=True, nargs=1)
@click.argument("file_patterns", type=str, required=True, nargs=-1)
@click.argument("out", type=str, required=True, nargs=1)
def _sample(sample_size, file_patterns, out):
    file_paths = []
    for file_pattern in file_patterns:
        file_paths += tf.gfile.Glob(file_pattern)

    random.shuffle(file_paths)

    # Try to read twice as many examples as requested from the files, reading
    # the files in a random order.
    buffer_size = int(2 * sample_size)
    examples = []
    for file_name in file_paths:
        for example in tf.python_io.tf_record_iterator(file_name):
            examples.append(example)
            if len(examples) == buffer_size:
                break
        if len(examples) == buffer_size:
            break

    if len(examples) < sample_size:
        tf.logging.warning(
            "Not enough examples to sample from. Found %i but requested %i.",
            len(examples), sample_size,
        )
        sampled_examples = examples
    else:
        sampled_examples = random.sample(examples, sample_size)

    with tf.python_io.TFRecordWriter(out) as record_writer:
        for example in sampled_examples:
            record_writer.write(example)

    print("Wrote %i examples to %s." % (len(sampled_examples), out))


@_cli.command(name="pp")
@click.argument("path", type=str, required=True, nargs=1)
def _pretty_print(path):
    """Format and print the contents of the tfrecord file to stdout."""
    for i, record in enumerate(tf.python_io.tf_record_iterator(path)):
        example = tf.train.Example()
        example.ParseFromString(record)
        print("Example %i\n--------" % i)
        _pretty_print_example(example)
        print("--------\n\n")


def _pretty_print_example(example):
    """Format and print an individual tensorflow example."""
    _print_field("Context", _get_string_feature(example, "context"))
    _print_field("Response", _get_string_feature(example, "response"))
    _print_extra_contexts(example)
    _print_other_features(example)


def _print_field(name, content, indent=False):
    indent_str = "\t" if indent else ""
    content = content.replace("\n", "\\n ")
    print("%s[%s]:" % (indent_str, name))
    print("%s\t%s" % (indent_str, content))


def _get_string_feature(example, feature_name):
    return example.features.feature[feature_name].bytes_list.value[0].decode(
        "utf-8")


def _print_extra_contexts(example):
    """Print the extra context features."""
    extra_contexts = []
    i = 0
    while True:
        feature_name = "context/{}".format(i)
        try:
            value = _get_string_feature(example, feature_name)
        except IndexError:
            break
        extra_contexts.append((feature_name, value))
        i += 1
    if not extra_contexts:
        return

    print("\nExtra Contexts:")
    for feature_name, value in reversed(extra_contexts):
        _print_field(feature_name, value, indent=True)


def _print_other_features(example):
    """Print the other features, which will depend on the dataset.

    For now, only support string features.
    """
    printed_header = False
    for feature_name, value in sorted(example.features.feature.items()):
        if (feature_name in {"context", "response"} or
                feature_name.startswith("context/")):
            continue
        if not printed_header:
            # Only print the header if there are other features in this
            # example.
            print("\nOther features:")

        printed_header = True
        _print_field(
            feature_name, value.bytes_list.value[0].decode("utf-8"),
            indent=True)


if __name__ == "__main__":
    if six.PY2:
        sys.stdout = codecs.getwriter("utf8")(sys.stdout)
    _cli()
