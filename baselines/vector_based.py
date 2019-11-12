"""Methods for conversational response ranking based on vector comparisons."""

import abc
import itertools
import shutil
import tempfile

import glog
import numpy as np
import tensorflow as tf
import tensorflow_hub
import tensorflow_text  # NOQA: required for PolyAI encoders.
import tf_sentencepiece  # NOQA: it is used when importing USE_QA.
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import bert.run_classifier
import bert.tokenization
from baselines import method


class Encoder(object):
    """A model that maps from text to dense vectors."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode_context(self, contexts):
        """Encode the given texts as vectors.

        Args:
            contexts: a list of N strings, to be encoded.

        Returns:
            an (N, d) numpy matrix of encodings.
        """
        pass

    def encode_response(self, responses):
        """Encode the given response texts as vectors.

        Args:
            responses: a list of N strings, to be encoded.

        Returns:
            an (N, d) numpy matrix of encodings.
        """
        # Default to using the context encoding.
        return self.encode_context(responses)


class TfHubEncoder(Encoder):
    """An encoder that is loaded as a module from tensorflow hub.

    The tensorflow hub module must take a vector of strings, and return
    a matrix of encodings.

    Args:
        uri: (string) the tensorflow hub URI for the model.
    """
    def __init__(self, uri):
        """Create a new `TfHubEncoder` object."""
        self._session = tf.Session(graph=tf.Graph())
        with self._session.graph.as_default():
            glog.info("Loading %s model from tensorflow hub", uri)
            embed_fn = tensorflow_hub.Module(uri)
            self._fed_texts = tf.placeholder(shape=[None], dtype=tf.string)
            self._context_embeddings = embed_fn(self._fed_texts)
            init_ops = (
                tf.global_variables_initializer(), tf.tables_initializer())
        glog.info("Initializing graph.")
        self._session.run(init_ops)

    def encode_context(self, contexts):
        """Encode the given texts."""
        return self._session.run(
            self._context_embeddings, {self._fed_texts: contexts})


class USEDualEncoder(Encoder):
    """A dual encoder following the USE_QA signatures.

    Args:
        uri: (string) the tensorflow hub URI for the model.
    """
    def __init__(self, uri):
        """Create a new `USEDualEncoder` object."""
        self._session = tf.Session(graph=tf.Graph())
        with self._session.graph.as_default():
            glog.info("Loading %s model from tensorflow hub", uri)
            embed_fn = tensorflow_hub.Module(uri)
            self._fed_texts = tf.placeholder(shape=[None], dtype=tf.string)
            self._context_embeddings = embed_fn(
                dict(input=self._fed_texts),
                signature="question_encoder",
                as_dict=True,
            )['outputs']
            empty_strings = tf.fill(
                tf.shape(self._fed_texts), ""
            )
            self._response_embeddings = embed_fn(
                dict(input=self._fed_texts, context=empty_strings),
                signature="response_encoder",
                as_dict=True,
            )['outputs']
            init_ops = (
                tf.global_variables_initializer(), tf.tables_initializer())
        glog.info("Initializing graph.")
        self._session.run(init_ops)

    def encode_context(self, contexts):
        """Encode the given texts as contexts."""
        return self._session.run(
            self._context_embeddings, {self._fed_texts: contexts})

    def encode_response(self, responses):
        """Encode the given texts as responses."""
        return self._session.run(
            self._response_embeddings, {self._fed_texts: responses})


class ConveRTEncoder(Encoder):
    """The ConveRT encoder.

    See https://github.com/PolyAI-LDN/polyai-models.

    Args:
        uri: (string) the tensorflow hub URI for the model.
    """
    def __init__(self, uri):
        """Create a new `ConveRTEncoder` object."""
        self._session = tf.Session(graph=tf.Graph())
        with self._session.graph.as_default():
            glog.info("Loading %s model from tensorflow hub", uri)
            embed_fn = tensorflow_hub.Module(uri)
            self._fed_texts = tf.placeholder(shape=[None], dtype=tf.string)
            self._context_embeddings = embed_fn(
                self._fed_texts, signature="encode_context")
            self._response_embeddings = embed_fn(
                self._fed_texts, signature="encode_response")
            init_ops = (
                tf.global_variables_initializer(), tf.tables_initializer())
        glog.info("Initializing graph.")
        self._session.run(init_ops)

    def encode_context(self, contexts):
        """Encode the given texts as contexts."""
        return self._session.run(
            self._context_embeddings, {self._fed_texts: contexts})

    def encode_response(self, responses):
        """Encode the given texts as responses."""
        return self._session.run(
            self._response_embeddings, {self._fed_texts: responses})


class BERTEncoder(Encoder):
    """The BERT encoder that is loaded as a module from tensorflow hub.

    This class tokenizes the input text using the bert tokenization
    library. The final encoding is computed as the sum of the token
    embeddings.

    Args:
        uri: (string) the tensorflow hub URI for the model.
    """
    def __init__(self, uri):
        """Create a new `BERTEncoder` object."""
        if not tf.test.is_gpu_available():
            glog.warning(
                "No GPU detected, BERT will run a lot slower than with a GPU.")

        self._session = tf.Session(graph=tf.Graph())
        with self._session.graph.as_default():
            glog.info("Loading %s model from tensorflow hub", uri)
            embed_fn = tensorflow_hub.Module(uri, trainable=False)
            self._tokenizer = self._create_tokenizer_from_hub_module(uri)
            self._input_ids = tf.placeholder(
                name="input_ids", shape=[None, None], dtype=tf.int32)
            self._input_mask = tf.placeholder(
                name="input_mask", shape=[None, None], dtype=tf.int32)
            self._segment_ids = tf.zeros_like(self._input_ids)
            bert_inputs = dict(
                input_ids=self._input_ids,
                input_mask=self._input_mask,
                segment_ids=self._segment_ids
            )

            embeddings = embed_fn(
                inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]
            mask = tf.expand_dims(
                tf.cast(self._input_mask, dtype=tf.float32), -1)
            self._embeddings = tf.reduce_sum(mask * embeddings, axis=1)

            init_ops = (
                tf.global_variables_initializer(), tf.tables_initializer())
        glog.info("Initializing graph.")
        self._session.run(init_ops)

    def encode_context(self, contexts):
        """Encode the given texts."""
        return self._session.run(self._embeddings, self._feed_dict(contexts))

    @staticmethod
    def _create_tokenizer_from_hub_module(uri):
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            bert_module = tensorflow_hub.Module(uri, trainable=False)
            tokenization_info = bert_module(
                signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run(
                    [
                        tokenization_info["vocab_file"],
                        tokenization_info["do_lower_case"]
                    ])

        return bert.tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

    def _feed_dict(self, texts, max_seq_len=128):
        """Create a feed dict for feeding the texts as input.

        This uses dynamic padding so that the maximum sequence length is the
        smaller of `max_seq_len` and the longest sequence actually found in the
        batch. (The code in `bert.run_classifier` always pads up to the maximum
        even if the examples in the batch are all shorter.)
        """
        all_ids = []
        for text in texts:
            tokens = ["[CLS]"] + self._tokenizer.tokenize(text)

            # Possibly truncate the tokens.
            tokens = tokens[:(max_seq_len - 1)]
            tokens.append("[SEP]")
            ids = self._tokenizer.convert_tokens_to_ids(tokens)
            all_ids.append(ids)

        max_seq_len = max(map(len, all_ids))

        input_ids = []
        input_mask = []
        for ids in all_ids:
            mask = [1] * len(ids)

            # Zero-pad up to the sequence length.
            while len(ids) < max_seq_len:
                ids.append(0)
                mask.append(0)

            input_ids.append(ids)
            input_mask.append(mask)

        return {self._input_ids: input_ids, self._input_mask: input_mask}


class VectorSimilarityMethod(method.BaselineMethod):
    """Ranks responses using cosine similarity of context & response vectors.

    Args:
        encoder: the `Encoder` object to use.
    """
    def __init__(self, encoder):
        """Create a new `VectorSimilarityMethod` object."""
        self._encoder = encoder

    def train(self, contexts, responses):
        """Train on the contexts and responses. Does nothing."""
        pass

    def rank_responses(self, contexts, responses):
        """Rank the responses for each context, using cosine similarity."""
        contexts_matrix = self._encoder.encode_context(contexts)
        responses_matrix = self._encoder.encode_response(responses)
        responses_matrix /= np.linalg.norm(
            responses_matrix, axis=1, keepdims=True)
        similarities = np.matmul(contexts_matrix, responses_matrix.T)
        return np.argmax(similarities, axis=1)


class VectorMappingMethod(method.BaselineMethod):
    """Applies a linear mapping to the response side and ranks with similarity.

    This learns a [dim, dim] weights matrix, and maps the response vector `x`
    to `x + weights.x`. The weights matrix is learned using gradient descent
    on the train set, and the dot product loss from
    https://arxiv.org/abs/1705.00652 . A grid search over hyper-parameters is
    performed, and the weights that get the best accuracy on the dev set are
    used.

    Args:
        encoder: the `Encoder` object to use.
        learning_rates: the learning rates to try in grid search.
        regularizers: the regularizers to try in grid search.
    """
    def __init__(
        self,
        encoder,
        learning_rates=(10.0, 3.0, 1.0, 0.3, 0.01),
        regularizers=(0, 0.1, 0.01, 0.001),
    ):
        """Create a new `VectorMappingMethod` object."""
        self._encoder = encoder
        self._learning_rates = learning_rates
        self._regularizers = regularizers

    def train(self, contexts, responses):
        """Train on the contexts and responses."""
        glog.info(
            "Training on %i contexts and responses.", len(contexts))
        (contexts_train, contexts_dev,
         responses_train, responses_dev
         ) = self._create_train_and_dev(contexts, responses)
        glog.info(
            "Created a training set of size %i, and a dev set of size %i.",
            contexts_train.shape[0], contexts_dev.shape[0])
        self._build_mapping_graph(
            contexts_train, contexts_dev,
            responses_train, responses_dev
        )
        self._grid_search()

    # Batch size to use when encoding texts.
    _ENCODING_BATCH_SIZE = 100
    _TRAIN_BATCH_SIZE = 256
    _MAX_EPOCHS = 100

    def _create_train_and_dev(self, contexts, responses):
        """Create a train and dev set of context and response vectors."""
        glog.info("Encoding the train set.")
        context_encodings = []
        response_encodings = []

        for i in tqdm(range(0, len(contexts), self._ENCODING_BATCH_SIZE)):
            contexts_batch = contexts[i:i + self._ENCODING_BATCH_SIZE]
            responses_batch = responses[i:i + self._ENCODING_BATCH_SIZE]
            context_encodings.append(
                self._encoder.encode_context(contexts_batch))
            response_encodings.append(
                self._encoder.encode_response(responses_batch))

        context_encodings = np.concatenate(
            context_encodings).astype(np.float32)
        response_encodings = np.concatenate(
            response_encodings).astype(np.float32)

        return train_test_split(
            context_encodings, response_encodings,
            test_size=0.2)

    def _build_mapping_graph(self,
                             contexts_train, contexts_dev,
                             responses_train, responses_dev):
        """Build the graph that applies a learned mapping to the vectors."""
        self._session = tf.Session(graph=tf.Graph())
        with self._session.graph.as_default():

            def read_batch(contexts, responses, batch_size):
                dataset = tf.data.Dataset.from_tensor_slices(
                    (contexts, responses))
                dataset = dataset.shuffle(batch_size * 8)
                dataset = dataset.batch(batch_size)
                return dataset.make_initializable_iterator()

            self._train_iterator = read_batch(
                contexts_train, responses_train,
                batch_size=self._TRAIN_BATCH_SIZE)
            self._dev_iterator = read_batch(
                contexts_dev, responses_dev,
                batch_size=100)

            (contexts_batch_train,
             responses_batch_train) = self._train_iterator.get_next()
            (contexts_batch_dev,
             responses_batch_dev) = self._dev_iterator.get_next()

            # Create the train op.
            self._regularizer = tf.placeholder(dtype=tf.float32, shape=None)
            self._create_train_op(
                self._compute_similarities(
                    contexts_batch_train, responses_batch_train,
                    is_train=True)
            )

            # Create the accuracy eval metric.
            dev_batch_size = tf.shape(contexts_batch_dev)[0]
            similarities = self._compute_similarities(
                contexts_batch_dev, responses_batch_dev,
                is_train=False)
            self._accuracy = tf.metrics.accuracy(
                labels=tf.range(dev_batch_size),
                predictions=tf.argmax(similarities, 1)
            )

            # Create the inference graph.
            encoding_dim = int(contexts_batch_train.shape[1])
            self._fed_context_encodings = tf.placeholder(
                dtype=tf.float32, shape=[None, encoding_dim]
            )
            self._fed_response_encodings = tf.placeholder(
                dtype=tf.float32, shape=[None, encoding_dim]
            )
            self._similarities = self._compute_similarities(
                self._fed_context_encodings,
                self._fed_response_encodings
            )

            self._local_init_op = tf.local_variables_initializer()
            self._reset_op = tf.global_variables_initializer()
            self._saver = tf.train.Saver(max_to_keep=1)

    def _compute_similarities(self, context_encodings, response_encodings,
                              is_train=False):
        """Compute the similarities between context and responses.

        Uses a learned mapping on the response side.
        """
        with tf.variable_scope("compute_similarities", reuse=(not is_train)):
            # Normalise the vectors so that the model is not dependent on
            # vector scaling.
            context_encodings = tf.nn.l2_normalize(context_encodings, 1)
            response_encodings = tf.nn.l2_normalize(response_encodings, 1)
            encoding_dim = int(context_encodings.shape[1])
            mapping_weights = tf.get_variable(
                "mapping_weights",
                dtype=tf.float32,
                shape=[encoding_dim, encoding_dim],
                initializer=tf.orthogonal_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(
                    self._regularizer),
            )
            residual_weight = tf.get_variable(
                "residual_weight",
                dtype=tf.float32,
                shape=[],
                initializer=tf.constant_initializer(1.0),
            )

            responses_mapped = tf.matmul(response_encodings, mapping_weights)
            responses_mapped += residual_weight * response_encodings

            return tf.matmul(
                context_encodings, responses_mapped,
                transpose_b=True)

    def _create_train_op(self, similarities):
        """Create the train op."""
        train_batch_size = tf.shape(similarities)[0]
        tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(
                tf.range(train_batch_size), train_batch_size),
            label_smoothing=0.2,
            logits=similarities,
            reduction=tf.losses.Reduction.MEAN
        )
        self._learning_rate = tf.placeholder(dtype=tf.float32, shape=None)
        self._train_op = tf.contrib.training.create_train_op(
            total_loss=tf.losses.get_total_loss(),
            optimizer=tf.train.AdagradOptimizer(
                learning_rate=self._learning_rate))

    def _grid_search(self):
        """Perform a grid search of training hyper-parameters.

        The model that does the best on the dev set will be stored.
        """
        save_path = tempfile.mkdtemp(prefix="VectorMappingMethod")

        def _compute_accuracy():
            self._session.run(self._local_init_op)
            self._session.run(self._dev_iterator.initializer)
            while True:
                try:
                    accuracy, _ = self._session.run(self._accuracy)
                except tf.errors.OutOfRangeError:
                    return accuracy

        best_accuracy, best_learning_rate, best_regularizer = None, None, None

        for learning_rate, regularizer in itertools.product(
                self._learning_rates, self._regularizers):
            # Train using this learning rate and regularizer.
            self._session.run(self._reset_op)
            best_accuracy_for_run = None
            epochs_since_improvement = 0
            epoch = 0
            step = 0
            glog.info(
                "\n\nTraining with learning_rate = %.5f, "
                "and regularizer = %.5f", learning_rate, regularizer)
            self._session.run(self._train_iterator.initializer)

            while epoch < self._MAX_EPOCHS:
                try:
                    loss = self._session.run(
                        self._train_op,
                        {self._learning_rate: learning_rate,
                         self._regularizer: regularizer})
                    step += 1

                except tf.errors.OutOfRangeError:
                    epoch += 1
                    accuracy = _compute_accuracy()
                    log_suffix = ""
                    self._session.run(self._train_iterator.initializer)

                    if best_accuracy is None or accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_learning_rate = learning_rate
                        best_regularizer = regularizer
                        self._saver.save(self._session, save_path)
                        log_suffix += "*"

                    if (best_accuracy_for_run is None
                            or accuracy > best_accuracy_for_run):
                        epochs_since_improvement = 0
                        best_accuracy_for_run = accuracy
                        log_suffix += "*"

                    glog.info(
                        "epoch %i: step: %i, loss: %.3f, "
                        "dev accuracy: %.2f%% %s",
                        epoch, step, loss, accuracy * 100, log_suffix)

                    epochs_since_improvement += 1
                    if epochs_since_improvement >= 10:
                        glog.info(
                            "No improvement for %i epochs, terminating run.",
                            epochs_since_improvement)
                        break

        glog.info(
            "Best accuracy found was %.2f%%, with learning_rate = %.5f and "
            "regularizer = %.5f.",
            best_accuracy * 100,
            best_learning_rate, best_regularizer)
        self._saver.restore(self._session, save_path)
        shutil.rmtree(save_path)

    def rank_responses(self, contexts, responses):
        """Rank the responses for each context."""
        similarities = self._session.run(
            self._similarities,
            {
                self._fed_context_encodings: self._encoder.encode_context(
                    contexts),
                self._fed_response_encodings: self._encoder.encode_response(
                    responses),
            }
        )
        return np.argmax(similarities, axis=1)
