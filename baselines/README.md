
* description of baselines
* intended to be relatively fast to run locally, to give an idea of the characteristics of the dataset, what a good accuracy would be. not intended to be highly competitive with e.g. models that use the entire training set.
* can train on 10 000 examples randomly sampled from the train set, and are evaluated on 500 batches of 100 examples in the test set


```
# Copy a random sample of the data locally.
mkdir data
gsutil cp ${DATADIR?}/train-00001-* data/
gsutil cp ${DATADIR?}/test-00001-* data/

```

For Amazon, you will need to copy two shards of the test set to get enough examples.

```
export TFHUB_CACHE_DIR=~/.tfhub_cache

python baselines/run_baseline.py  \
  --method BM25 \
  --train_dataset data/train-* \
  --test_dataset data/test-*
```
