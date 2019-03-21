# Baseline methods for conversational response selection

This directory provides several baseline methods for conversational response selection. These help to benchmark the performance on the various datasets.

Baselines are intended to be relatively fast to run locally, and are not intended to be highly competitive with state of the art methods. As such they are limited to using a small portion of the training set, typically 10 thousand randomly sampled examples.


## Keyword-based

The keyword-based baselines use keyword similarity metrics to rank responses given a context. The `TF_IDF` method computes inverse document frequency statistics on the training set. Responses are scored using their [tf-idf cosine similarity](https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/) to the context.

The `BM25` method builds on top of the tf-idf similarity, applying an adjustment to the term weights. See [Okapi BM25: a non-binary model](http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html) for further discussion of the approach.


## Vector-based

Vector-based methods use publicly available neural network embedding models to embed contexts and responses into a vector space. The currently implemented models are:

* [USE](https://tfhub.dev/google/universal-sentence-encoder/2) - the universal sentence encoder
* [USE_LARGE](https://tfhub.dev/google/universal-sentence-encoder-large/3) - a larger version of the universal sentence encoder
* [ELMO](https://tfhub.dev/google/elmo/1) - the Embeddings from Language Models approach

all of which are loaded from Tensorflow Hub.


There are two vector-based baseline method, one for each of the above models. The `SIM` method ranks responses according to their cosine similarity with the context vector. This method does not use the training set at all. The `MAP` method learns a linear mapping on top of the response vector. The final score of a response with vector ![https://latex.codecogs.com/svg.latex?\mathbf{y}] given a context with vector ![https://latex.codecogs.com/svg.latex?\mathbf{x}] is:

![https://latex.codecogs.com/gif.latex?\mathbf{x}&space;\cdot&space;\frac{&space;\mathbf{y}'}{&space;\left&space;\|&space;\mathbf{y}'&space;\right&space;\|}]

where

![https://latex.codecogs.com/gif.latex?\mathbf{y}'&space;=(W&space;&plus;&space;\alpha&space;I)&space;\cdot&space;\mathbf{y}]



# Running the baselines

## Copy the data locally


```
# Copy a random sample of the data locally.
mkdir data
gsutil cp ${DATADIR?}/train-00001-* data/
gsutil cp ${DATADIR?}/test-00001-* data/

```

For Amazon, you will need to copy two shards of the test set to get enough examples.

## Run the baselines

```
export TFHUB_CACHE_DIR=~/.tfhub_cache

python baselines/run_baseline.py  \
  --method BM25 \
  --train_dataset data/train-* \
  --test_dataset data/test-*
```

[`run_baselines.sh`](run_baselines.sh)
