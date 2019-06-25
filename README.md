[![CircleCI](https://circleci.com/gh/PolyAI-LDN/conversational-datasets.svg?style=svg&circle-token=25d37b8026cb0c81306db68d098703f81dd74da9)](https://circleci.com/gh/PolyAI-LDN/conversational-datasets)

[![PolyAI](polyai-logo.png)](https://poly-ai.com/)

# conversational-datasets

*A collection of large datasets for conversational response selection.*

This repository provides tools to create reproducible datasets for training and evaluating models of conversational response. This includes:

* [Reddit](reddit) - 3.7 billion comments structured in threaded conversations
* [OpenSubtitles](opensubtitles) - over 400 million lines from movie and television subtitles (available in English and other languages)
* [Amazon QA](amazon_qa) - over 3.6 million question-response pairs in the context of Amazon products

Machine learning methods work best with large datasets such as these. At PolyAI we train models of conversational response on huge conversational datasets and then adapt these models to domain-specific tasks in conversational AI. This general approach of pre-training large models on huge datasets has long been popular in the image community and is now taking off in the NLP community.

Rather than providing the raw processed data, we provide scripts and instructions to generate the data yourself. This allows you to view and potentially manipulate the pre-processing and filtering. The instructions define standard datasets, with deterministic train/test splits, which can be used to define reproducible evaluations in research papers.

## Datasets

Each dataset has its own directory, which contains a dataflow script, instructions for running it, and unit tests.

|                                	|                                     	| Train set size 	| Test set size 	|
|--------------------------------	|-------------------------------------	|----------------	|---------------	|
| [Reddit](reddit)               	| 2015 - 2019                         	| 654 million    	| 72 million    	|
| [OpenSubtitles](opensubtitles) 	| English (other languages available) 	| 286 million    	| 33 million    	|
| [Amazon QA](amazon_qa)         	| -                                   	| 3 million      	| 0.3 million   	|

Note that these are the dataset sizes after filtering and other processing. For instance, the Reddit dataset is based on a raw database of 3.7 billion comments, but consists of 726 million examples because the script filters out long comments, short comments, uninformative comments (such as `'[deleted]'`, and comments with no replies.


## Benchmarks

Benchmark results for each of the datasets can be found in [`BENCHMARKS.md`](BENCHMARKS.md).

## Conversational Dataset Format

This repo contains scripts for creating datasets in a standard format -
any dataset in this format is referred to elsewhere as simply a
*conversational dataset*.


Datasets are stored either as:

* JSON text files, with one example per line
* or as [Tensorflow record files](`https://www.tensorflow.org/tutorials/load_data/tf_records`) containing serialized [tensorflow example](https://www.tensorflow.org/tutorials/load_data/tf_records#data_types_for_tfexample) protocol buffers.


The training set is stored as one collection of examples, and
the test set as another. Examples are shuffled randomly (and not necessarily reproducibly) among the files. The train/test split is always deterministic, so that whenever the dataset is generated, the same train/test split is created.

Each example contains a conversational context and a response that goes with that context. For example:

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
  back in time through the conversation. They are named in reverse order so that `context/i` always refers to the `i^th` most recent extra context, so that no padding needs to be done, and datasets with different numbers of extra contexts can be mixed.

Depending on the dataset, there may be some extra features also included in
each example. For instance, in Reddit the author of the context and response are
identified using additional features.

### Reading conversational datasets

#### JSON format

For use outside of tensorflow, the JSON format may be preferable. To get JSON format datasets, use `--dataset_format JSON` in the dataset's `create_data.py` script. Each line will contain a single JSON object.

Below is some example python code for reading a JSON format dataset.

```python
import json
from glob import glob


for file_name in glob("dataset/train/*.json"):
    for line in open(file_name):
        example = json.loads(line)
        # You can now access:
        #   example['context']
        #   example['response']
        #   example['context/0'] etc.
```

#### Tensorflow format

The [`tools/tfrutil.py`](tools/tfrutil.py) and [`baselines/run_baseline.py`](baselines/run_baseline.py) scripts demonstrate how to read a Tensorflow example format conversational dataset in Python, using functions from the tensorflow library.

You can use [`tools/tfrutil.py`](tools/tfrutil.py) to compute the number of examples in a tensorflow record file:

```
$ python tools/tfrutil.py size data/reddit-test
726158
```

It can also be used to display the examples in a readable format:

```
$ python tools/tfrutil.py pp data/reddit-test

Example 0
--------
[Context]:
	Airplane? What is it?
[Response]:
	Airplane! The movie. It's an amazing parody of plane movies which sounds terrible but it is actually 10/10.

Extra Contexts:
	[context/2]:
		Unfortunately, they all had the fish for dinner.
	[context/1]:
		This is some sort of reference? I don't get it.
	[context/0]:
		Airplane. Drop everything and watch it right now

Other features:
	[context_author]:
		Doctor_Insano_MD
	[response_author]:
		ThegreatandpowerfulR
	[subreddit]:
		todayilearned
	[thread_id]:
		41ar0l
--------
...
```

Below is some example tensorflow code for reading a conversational dataset
into a tensorflow graph:

```python

num_extra_contexts = 10
batch_size = 100
pattern = "gs://your-bucket/dataset/train-*.tfrecord"

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

Conversational datasets are created using [Apache Beam pipeline](https://beam.apache.org/) scripts, run on [Google Dataflow](https://cloud.google.com/dataflow/). This parallelises the data processing pipeline across many worker machines. Apache Beam requires python 2.7, so you will need to set up a python 2.7 virtual environment:

```bash
python2.7 -m virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```

The Dataflow scripts write conversational datasets to Google cloud storage, so you will need to [create a bucket](https://cloud.google.com/storage/docs/creating-buckets) to save the dataset to.

Dataflow will run workers on multiple Compute Engine instances, so make sure you have a sufficient [quota](https://cloud.google.com/dataflow/quotas) of `n1-standard-1` machines. The READMEs for individual datasets give an idea of how many workers are required, and how long each dataflow job should take.

Lastly, you will need to [set up authentication](https://cloud.google.com/docs/authentication/getting-started) by creating a service account with access to Dataflow and Cloud Storage, and set `GOOGLE_APPLICATION_CREDENTIALS`:

```bash
export GOOGLE_APPLICATION_CREDENTIALS={{json file key location}}
```

This should be enough to follow the instructions for creating each individual dataset.


## Evaluation

Of course you may evaluate your models in any way you like.
However, when publishing results, we encourage you to include the
1-of-100 ranking accuracy, which is becoming a research community standard.

The 1-of-100 ranking accuracy is a *Recall@k* metric. In general *Recall@k*
takes *N* responses to the given conversational context, where only one response is relevant. It indicates whether the relevant response occurs in the top *k* ranked candidate responses.
The 1-of-100 metric is obtained when *k=1* and *N=100*.
This effectively means that, for each query, we indicate if the correct response is the top ranked response among 100 candidates. The final score is the average across all queries.

The 1-of-100 metric is computed using random batches of 100 examples so that the responses from other examples in the batch are used as random negative candidates. This allows for efficiently computing the metric across many examples in batches. While it is not guaranteed that the random negatives will indeed be 'true' negatives, the 1-of-100 metric still provides a useful evaluation signal that correlates with downstream tasks.

The following tensorflow code shows how this metric can be computed for a dot-product style encoder model, where the score for each context and response is a dot product between corresponding vectors:

```python
# Encode the contexts and responses as vectors using tensorflow ops.
# The following are both [100, encoding_size] matrices.
context_encodings = _encode_contexts(tensor_dict['context'])
response_encodings = _encode_responses(tensor_dict['response'])

scores = tf.matmul(
  context_encodings, response_encodings,
  transpose_b=True)  # A [100, 100] matrix.

batch_size = tf.shape(context_encodings)[0]

accuracy_1_of_100 = tf.metrics.accuracy(
  labels=tf.range(batch_size),
  predictions=tf.argmax(scores, 1)
)
```

See also the [baselines](baselines) for example code computing the 1-of-100 metric.

Many studies have used *Recall@k* in the context of retrieval-based dialogue, including the following papers:

* [*The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems*](https://aclweb.org/anthology/W15-4640), Lowe et al. SIGDIAL 2015.

* [*Neural Utterance Ranking Model for Conversational Dialogue Systems*](https://www.aclweb.org/anthology/W16-3648), Inaba and Takahashi. SIGDIAL 2016.

* [*Strategy and Policy Learning for Non-task-oriented Conversational Systems*](https://aclweb.org/anthology/W16-3649), Yu et al. SIGDIAL 2016.

* [*Training End-to-End Dialogue Systems with the Ubuntu Dialogue Corpus*](http://dad.uni-bielefeld.de/index.php/dad/article/view/3698), Lowe et al. Dialogue and Discourse 2017.

* [*Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-based Chatbots*](https://aclweb.org/anthology/P17-1046), Wu et al. ACL 2017.

* [*Improving Response Selection in Multi-turn Dialogue Systems by Incorporating Domain Knowledge*](http://aclweb.org/anthology/K18-1048),
Chaudhuri et al. CoNLL 2018.

* [*Data Augmentation for Neural Online Chats Response Selection*](https://aclweb.org/anthology/W18-5708), Du and Black. SCAI 2018.

* [*Customized Nonlinear Bandits for Online Response Selection in Neural Conversational Models*](https://arxiv.org/pdf/1711.08493.pdf), Liu et al. AAAI 2018.

* [*DSTC7 task 1: Noetic end-to-end response selection*](https://aclweb.org/anthology/W18-5708), Gunasekara et al. 2019.

* [*Multi-representation Fusion Network for Multi-Turn Response Selection in Retrieval-based Chatbots*](https://dl.acm.org/citation.cfm?doid=3289600.3290985), Tao et al. WSDM 2019.

* [*Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network*](http://aclweb.org/anthology/P18-1103), Zhou et al. ACL 2018.

The following papers use the 1-of-100 ranking accuracy in particular:

* [*Conversational Contextual Cues: The Case of Personalization  and History  for  Response  Ranking.*](http://arxiv.org/abs/1606.00372), Al-Rfou et al. arXiv pre-print 2016.

* [*Efficient Natural Language Response Suggestion for Smart Reply*](http://arxiv.org/abs/1705.00652), Henderson et al. arXiv pre-print 2017.

* [*Question-Answer Selection in User to User Marketplace Conversations*](https://arxiv.org/pdf/1802.01766.pdf), Kumar et al. IWSDS 2018.

* [*Universal Sentence Encoder*](https://arxiv.org/abs/1803.11175), Cer et al. arXiv pre-print 2018.

* [*Learning  Semantic  Textual  Similarity  from  Conversations.*](http://aclweb.org/anthology/W18-3022). Yang et al. Workshop on Representation Learning for NLP 2018.



## Citations

When using these datasets in your work, please cite our paper, [A Repository of Conversational Datasets](https://arxiv.org/abs/1904.06472):

```bibtex
@inproceedings{Henderson2019,
    author      = {Matthew Henderson and Pawe{\l} Budzianowski and I{\~{n}}igo Casanueva and Sam Coope and Daniela Gerz and Girish Kumar and Nikola Mrk{\v{s}}i\'c and Georgios Spithourakis and Pei-Hao Su and Ivan Vulic and Tsung-Hsien Wen},
    title       = {A Repository of Conversational Datasets},
    year        = {2019},
    month       = {jul},
    note        = {Data available at github.com/PolyAI-LDN/conversational-datasets},
    url         = {https://arxiv.org/abs/1904.06472},
    booktitle   = {Proceedings of the Workshop on {NLP} for Conversational {AI}},
}

```

## Contributing

We happily accept contributions in the form of pull requests.
Each pull request is tested in CircleCI - it is first linted with `flake8`, and then the unit tests are run. In particular we would be interested in:

* new datasets
* adaptations to the scripts so that they work better in your environment (e.g. other Apache Beam runners, other cloud storage solutions, other example formats)
* results from your methods in the benchmarks [the benchmarks page](BENCHMARKS.md).
* code for new baselines and improvements to existing baselines
