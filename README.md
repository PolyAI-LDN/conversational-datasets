[![CircleCI](https://circleci.com/gh/PolyAI-LDN/conversational-datasets.svg?style=svg&circle-token=25d37b8026cb0c81306db68d098703f81dd74da9)](https://circleci.com/gh/PolyAI-LDN/conversational-datasets)

[![PolyAI](polyai-logo.png)](https://poly-ai.com/)

# conversational-datasets

why scripts? deterministic train/test split.


## Conversational Dataset Format




```python

num_extra_contexts = 10
batch_size = 100
pattern = "gs://your-bucket/dataset/train-*.tfrecords"

if not tf.gfile.Glob(pattern):
    raise ValueError("No files matched pattern " + pattern)

dataset = tf.data.Dataset.list_files(pattern)
dataset = dataset.apply(
    tf.contrib.data.parallel_interleave(
        lambda file: tf.data.TFRecordDataset(file),
        cycle_length=8))
dataset = dataset.apply(
    tf.data.experimental.shuffle_and_repeat(
        buffer_size=8 * batch_size))
dataset = dataset.batch(batch_size)

def _parse_function(serialized_examples):
    parse_spec = {
        "context": tf.FixedLenFeature([], tf.string),
        "response": tf.FixedLenFeature([], tf.string)
    }
    parse_spec.update({
        "context/{}".format(i): tf.FixedLenFeature(
            [], tf.string, default_value="")
        for i in range(num_extra_contexts)
    })
    return tf.parse_example(serialized_examples, parse_spec)

dataset = dataset.map(_parse_function, num_parallel_calls=8)
dataset = dataset.prefetch(8)
iterator = dataset.make_one_shot_iterator()
tensor_dict = iterator.get_next()

# The tensorflow graph can now access
# tensor_dict["context"], tensor_dict["response"] etc.
# as batches of string features (unicode bytes).
```

## Getting Started

```
python2.7 -m virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```


## Datasets

### Reddit

## Evaluation

## Citations

## Contributing
