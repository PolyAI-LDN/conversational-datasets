"""Tests for validation scores used during mapping grid search."""

import os
import shutil
import tempfile
import unittest

import mock
import tensorflow as tf

from baselines import vector_based


class GridSearchMetricTest(unittest.TestCase):
    def test_checkpoint_includes_the_final_validation_batch(self):
        directory = tempfile.mkdtemp()
        checkpoint = os.path.join(directory, "model")
        os.mkdir(checkpoint)
        try:
            with tf.Session(graph=tf.Graph()) as session:
                with session.graph.as_default():
                    model = vector_based.VectorMappingMethod(
                        None, learning_rates=[0., 1.], regularizers=[0.])
                    model._MAX_EPOCHS = 1
                    model._session = session
                    model._learning_rate = tf.placeholder(tf.float32, [])
                    model._regularizer = tf.placeholder(tf.float32, [])
                    weight = tf.get_variable("weight", initializer=0.)
                    model._train_iterator = tf.data.Dataset.from_tensors(
                        0).make_initializable_iterator()
                    model._dev_iterator = tf.data.Dataset.from_tensors(
                        tf.constant([1])).make_initializable_iterator()
                    train_token = model._train_iterator.get_next()
                    with tf.control_dependencies([train_token]):
                        model._train_op = tf.assign(
                            weight, model._learning_rate)
                    labels = model._dev_iterator.get_next()
                    predictions = tf.fill(
                        tf.shape(labels), tf.cast(weight, tf.int32))
                    with tf.variable_scope("validation"):
                        value, _ = tf.metrics.accuracy(
                            labels, predictions, name="accuracy")
                    # Force one valid executor order: read the old value
                    # before updating it with this batch. These are real
                    # streaming metric variables, shared by both operations.
                    with tf.variable_scope("validation", reuse=True):
                        with tf.control_dependencies([value]):
                            _, update = tf.metrics.accuracy(
                                labels, predictions, name="accuracy")
                    model._accuracy = (value, update)
                    model._local_init_op = tf.local_variables_initializer()
                    model._reset_op = tf.global_variables_initializer()
                    model._saver = tf.train.Saver(var_list=[weight])
                with mock.patch(
                        "baselines.vector_based.tempfile.mkdtemp",
                        return_value=checkpoint):
                    model._grid_search()
                # Candidate 0 predicts the wrong label; candidate 1 is correct.
                # The restored checkpoint must belong to candidate 1.
                self.assertEqual(session.run(weight), 1.)
        finally:
            shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()
