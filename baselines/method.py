"""Abstract class to define a baseline response selection method."""

import abc


class BaselineMethod(object):
    """Abstract class to define a baseline response selection method."""
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
