"""Baseline response ranking methods using keyword matching."""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import (HashingVectorizer,
                                             TfidfTransformer,
                                             _document_frequency)
from sklearn.utils.testing import ignore_warnings

from baselines import method


class BM25Method(method.BaselineMethod):
    """BM25 baseline, using weighted keyword matching.

    Adapted from https://github.com/arosh/BM25Transformer/blob/master/bm25.py
    see Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html

    Args:
        k1: float, optional (default=2.0)
        b: float, optional (default=0.75)

    """
    def __init__(self, k1=2.0, b=0.75):
        """Create a new `BM25Method` object."""
        self._k1 = k1
        self._b = b

    def train(self, contexts, responses):
        """Fit the tf-idf transform and compute idf statistics."""
        with ignore_warnings():
            # Ignore deprecated `non_negative` warning.
            self._vectorizer = HashingVectorizer(non_negative=True)
        self._tfidf_transform = TfidfTransformer()
        count_matrix = self._tfidf_transform.fit_transform(
            self._vectorizer.transform(contexts + responses))
        n_samples, n_features = count_matrix.shape
        df = _document_frequency(count_matrix)
        idf = np.log((n_samples - df + 0.5) / (df + 0.5))
        self._idf_diag = sp.spdiags(
            idf, diags=0, m=n_features, n=n_features
        )
        document_lengths = count_matrix.sum(axis=1)
        self._average_document_length = np.mean(document_lengths)
        print(self._average_document_length)

    def _vectorize(self, strings):
        """Vectorize the given strings."""
        with ignore_warnings():
            # Ignore deprecated `non_negative` warning.
            tf_idf_vectors = self._tfidf_transform.transform(
                self._vectorizer.transform(strings))
        tf_idf_vectors = sp.csr_matrix(
            tf_idf_vectors, dtype=np.float64, copy=True)

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        document_lengths = tf_idf_vectors.sum(axis=1)

        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        num_terms = tf_idf_vectors.indptr[1:] - tf_idf_vectors.indptr[0:-1]

        # In each row, repeat `document_lengths` for `num_terms` times
        # Shape is (sum(num_terms), )
        rep = np.repeat(np.asarray(document_lengths), num_terms)

        # Compute BM25 score only for non-zero elements
        data = tf_idf_vectors.data * (self._k1 + 1) / (
            tf_idf_vectors.data + self._k1 * (
                1 - self._b + self._b * rep / self._average_document_length))

        vectors = sp.csr_matrix(
            (data, tf_idf_vectors.indices, tf_idf_vectors.indptr),
            shape=tf_idf_vectors.shape)
        vectors = vectors * self._idf_diag

        return vectors

    def rank_responses(self, contexts, responses):
        """Rank the responses for each context."""
        contexts_matrix = self._vectorize(contexts)
        responses_matrix = self._vectorize(responses)
        similarities = contexts_matrix.dot(responses_matrix.T).toarray()
        return np.argmax(similarities, axis=1)


class TfIdfMethod(method.BaselineMethod):
    """TF-IDF baseline.

    This hashes words to sparse IDs, and then computes tf-idf statistics for
    these hashed IDs. As a result, no words are considered out-of-vocabulary.
    """
    def train(self, contexts, responses):
        """Fit the tf-idf transform and compute idf statistics."""
        self._vectorizer = HashingVectorizer()
        self._tfidf_transform = TfidfTransformer()
        self._tfidf_transform.fit(
            self._vectorizer.transform(contexts + responses))

    def _vectorize(self, strings):
        """Vectorize the given strings."""
        tf_idf_vectors = self._tfidf_transform.transform(
            self._vectorizer.transform(strings))
        return sp.csr_matrix(
            tf_idf_vectors, dtype=np.float64, copy=True)

    def rank_responses(self, contexts, responses):
        """Rank the responses for each context."""
        contexts_matrix = self._vectorize(contexts)
        responses_matrix = self._vectorize(responses)
        similarities = contexts_matrix.dot(responses_matrix.T).toarray()
        return np.argmax(similarities, axis=1)
