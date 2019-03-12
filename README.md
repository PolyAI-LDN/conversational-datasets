[![CircleCI](https://circleci.com/gh/PolyAI-LDN/conversational-datasets.svg?style=svg&circle-token=25d37b8026cb0c81306db68d098703f81dd74da9)](https://circleci.com/gh/PolyAI-LDN/conversational-datasets)

[![PolyAI](polyai-logo.png)](https://poly-ai.com/)

# conversational-datasets

why scripts? deterministic train/test split.


## Conversational Dataset Format

This repo contains scripts for creating datasets in a standard format -
any dataset in this format is referred to elsewhere as simply a
*conversational dataset*.

Datasets are stored as [tensorflow record files](`https://www.tensorflow.org/tutorials/load_data/tf_records`) containing serialized [tensorflow example](https://www.tensorflow.org/tutorials/load_data/tf_records#data_types_for_tfexample) protocol buffers.
The training set is stored as one collection of tensorflow record files, and
the test set as another.

Each tensorflow example contains a conversational context, and a response that goes with the context. For example:

```javascript
{
  'context/1': "Hello, how are you?",
  'context/0': "I am fine. And you?",
  'context': "Great. What do you think of the weather?",
  'response': "It doesn't feel like February."
}
```

Explicitly, each example contains a number of string features:

* A `context` feature, the most recent text in the conversational context
* A `response` feature, the text that is in direct response to the `context`.
* A number of *extra context features*, `context/0`, `context/1` etc. going
  back in time through the conversation. They are named in reverse order so that `context/i` always refers to the `i^th` most recent extra context, so no padding needs to be done, and so datasets with different numbers of extra contexts can be mixed.

Depending on the dataset, there may be some extra features also included in
each example. E.g. in Reddit, the author of the context and response are
identified using additional features.

### Reading conversational datasets

The [`tools/tfrutil.py`](tools/tfrutil.py) script demonstrates how to
read a conversational dataset in Python, using tensorflow functions.

Below is some example tensorflow code for reading a conversational dataset
into a tensorflow graph:

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

Conversational datasets are created using [Apache Beam pipeline](https://beam.apache.org/) scripts, run on [Google Dataflow](https://cloud.google.com/dataflow/). Apache Beam requires python 2.7, so set up a python 2.7 virtual environment:

```
python2.7 -m virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```

The Dataflow scripts write conversational datasets to Google cloud storage, so you will need to [create a bucket](https://cloud.google.com/storage/docs/creating-buckets) to save the dataset to.

Lastly, you will need to [set up authentication](
https://cloud.google.com/docs/authentication/getting-started) by creating a service account with access to Dataflow and Cloud Storage, and set `GOOGLE_APPLICATION_CREDENTIALS`:

```
export GOOGLE_APPLICATION_CREDENTIALS={{json file key location}}
```

This should be enough to follow the instructions for creating each individual dataset.

## Datasets

### Reddit

### OpenSubtitles

### Amazon QA

## Evaluation

We rely on a standard IR evaluation measure used in prior work on retrieval-based dialogue \cite{Lowe:2017dd,Zhou:2018acl,Chaudhuri:2018conll}: \textit{Recall@k}. Given a set of $N$ responses to the given input/query, where only one response is relevant, it indicates whether the relevant response occurs in the top $k$ ranked candidate responses. We refer to this evaluation measure as $\mathbf{R}_{N}@k$, and set $N=100; k=1$: $\mathbf{R}_{100}@1$. This effectively means that for each query, we indicate if the correct response is the top ranked response between 100 candidates. The final score is the average across all queries.

following also:

Matthew Henderson, Rami Al-Rfou, Brian Strope, Yun-Hsuan Sung, László Lukács, Ruiqi Guo, Sanjiv Ku-mar, Balint Miklos, and Ray Kurzweil. 2017.  Effi-cient natural language response suggestion for smartreply.CoRR, abs/1705.00652

using reddit:

Daniel  Cer,  Yinfei  Yang,  Sheng-yi  Kong,  Nan  Hua,Nicole Limtiaco, Rhomni St. John, Noah Constant,Mario  Guajardo-Cespedes,  Steve  Yuan,  Chris  Tar,Yun-Hsuan Sung, Brian Strope, and Ray Kurzweil.2018.Universal   sentence   encoder.CoRR,abs/1803.11175.

Rami  Al-Rfou,  Marc  Pickett,  Javier  Snaider,  Yun-Hsuan Sung, Brian Strope, and Ray Kurzweil. 2016.Conversational contextual cues: The case of person-alization  and  history  for  response  ranking.CoRR,abs/1606.00372

Yinfei Yang, Steve Yuan, Daniel Cer, Sheng-Yi Kong,Noah Constant, Petr Pilar, Heming Ge, Yun-hsuanSung, Brian Strope, and Ray Kurzweil. 2018. Learn-ing  semantic  textual  similarity  from  conversations.InProceedings of The 3rd Workshop on Representa-tion Learning for NLP, pages 164–174

## Citations

## Contributing
