"""Baseline response ranking method using keyword matching (TF-IDF and BM25)"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import (TfidfVectorizer,
                                             _document_frequency)

from baselines import method


class TfIdfMethod(method.BaselineMethod):
    """TF-IDF and BM25 baselines, using weighted keyword matching.

    Adapted from https://github.com/arosh/BM25Transformer/blob/master/bm25.py
    see Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html

    Args:
        apply_bm25_transform: boolean - whether to apply the bm25
            transformation on top of tf-idf. If False, this is just the TF-IDF
            baseline method.
        k1: float, optional (default=2.0)
        b: float, optional (default=0.75)

    """
    def __init__(self, apply_bm25_transform=False, k1=2.0, b=0.75):
        """Create a new `TfIdfMethod` object."""
        self._apply_bm25_transform = apply_bm25_transform
        self._k1 = k1
        self._b = b

    def train(self, contexts, responses):
        """Fit the tf-idf transform and compute idf statistics."""
        self._vectorizer = TfidfVectorizer()
        count_matrix = self._vectorizer.fit_transform(contexts + responses)
        if not self._apply_bm25_transform:
            return

        n_samples, n_features = count_matrix.shape
        df = _document_frequency(count_matrix)
        idf = np.log((n_samples - df + 0.5) / (df + 0.5))
        self._idf_diag = sp.spdiags(
            idf, diags=0, m=n_features, n=n_features
        )
        document_lengths = count_matrix.sum(axis=1)
        self._average_document_length = np.mean(document_lengths)

    def _vectorize(self, strings):
        """Vectorize the given strings."""
        tf_idf_vectors = self._vectorizer.transform(strings)
        tf_idf_vectors = sp.csr_matrix(
            tf_idf_vectors, dtype=np.float64, copy=True)

        if not self._apply_bm25_transform:
            return tf_idf_vectors

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
