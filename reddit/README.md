# Reddit data

[Reddit](https://www.reddit.com/) is an American social news aggregation website, where users can post links, and take part in discussions on these posts. These threaded discussions provide a large corpus, which is converted into a conversational dataset using the tools in this directory.

Each reddit thread is used to generate a set of examples. Each response comment generates an example, where the context is the linear path of comments that the comment is in response to. If the comment or its direct parent has more than 128 characters, or fewer than 9 characters, then the example is filtered out.

Further back contexts, from the comment's parent's parent etc., are stored as extra context features. Their texts are trimmed to be at most 128 characters in length, without splitting apart words. This helps to bound the size of an individual example.

The train / test split is deterministic based on the thread ID. As long as all the input to the script is held constant (the input tables, filtering thresholds etc.), the resulting datasets should be identical.


## Statistics

Below are some statistics of the dataset generated using the comments from
up to 2019 (`TABLE_REGEX="^201[5678]_[01][0-9]$")`):


* Number of comments: 3,680,746,776
* Number of threads: 256,095,216
* Number of tensorflow examples: 727,013,715
* Train set size: 654,396,778
* Test set size: 72,616,937

Typical metrics for the Dataflow job:

* Total vCPU time:  625.507 vCPU hr
* Total memory time: 322.5 GB hr
* Total persistent disk time: 156,376.805 GB hr
* Elapsed time: 1h 38m 409 workers)
* Estimated cost: 44 USD

# Create the conversational dataset

Below are instructions for how to generate the reddit dataset.

## Create the BigQuery input table

Reddit comment data is stored as a public BigQuery dataset, partitioned into months: [`fh-bigquery:reddit_comments.YYYY_MM`](https://console.cloud.google.com/bigquery?p=fh-bigquery&d=reddit_comments&page=dataset). The first step in creating the dataset is to create a single table that contains all the comment data to include.

First, [install the bq command-line tool](https://cloud.google.com/bigquery/docs/bq-command-line-tool).

Ensure you have a BigQuery dataset to write the table to:

```bash
DATASET="data"
bq mk --dataset ${DATASET?}
```

Write a new table by querying the public reddit data:

```bash
TABLE=reddit

# For all data up to 2019.
TABLE_REGEX="^201[5678]_[01][0-9]$"

QUERY="SELECT * \
  FROM TABLE_QUERY(\
  [fh-bigquery:reddit_comments], \
  \"REGEXP_MATCH(table_id, '${TABLE_REGEX?}')\" )"

# Run the query.
echo "${QUERY?}" | bq query \
  --n 0 \
  --batch --allow_large_results \
  --destination_table ${DATASET?}.${TABLE?} \
  --use_legacy_sql=true
```

## Run the dataflow script

[`create_data.py`](create_data.py) is a [Google Dataflow](https://cloud.google.com/dataflow/) script that reads the input BigQuery table and saves the dataset to Google Cloud Storage.


Now you can run the Dataflow script:

```bash
PROJECT="your-google-cloud-project"
BUCKET="your-bucket"

DATADIR="gs://${BUCKET?}/reddit/$(date +"%Y%m%d")"

# The below uses values of $DATASET and $TABLE set
# in the previous section.

python reddit/create_data.py \
  --output_dir ${DATADIR?} \
  --reddit_table ${PROJECT?}:${DATASET?}.${TABLE?} \
  --runner DataflowRunner \
  --temp_location ${DATADIR?}/temp \
  --staging_location ${DATADIR?}/staging \
  --project ${PROJECT?} \
  --dataset_format TF
```

You may use `--dataset_format JSON` to output JSON examples, rather than serialized Tensorflow examples in TFRecords.

Once the above is running, you can continue to monitor it in the terminal, or quit the process and follow the running job on the
[dataflow admin page](https://console.cloud.google.com/dataflow).

Please confirm that the statistics reported on the dataflow job page agree with the statistics reported above, to ensure you have a correct version of the dataset.

The dataset will be saved in the `$DATADIR` directory, as sharded train and test sets- `gs://your-bucket/reddit/YYYYMMDD/train-*-of-01000.tfrecord` and
`gs://your-bucket/reddit/YYYYMMDD/test-*-of-00100.tfrecord`. (Files will be stored as `.json` shards when using `--dataset_format JSON`.)

For tensorflow format, you can use [`tools/tfrutil.py`](/tools/tfrutil.py) to inspect the files. For example:

```bash
python tools/tfrutil.py pp ${DATADIR?}/train-00999-of-01000.tfrecord
```

(It may be faster to copy the tfrecord file locally first.) This will print examples like:

```
[Context]:
	"Learning to learn", using deep learning to design the architecture of another deep network: https://arxiv.org/abs/1606.04474
[Response]:
	using deep learning with SGD to design the learning algorithms of another deep network   *

Extra Contexts:
	[context/2]:
		Could someone there post a summary of the insightful moments.
	[context/1]:
		Basically L2L is the new deep learning.
	[context/0]:
		What's "L2L" mean?

Other features:
	[context_author]:
		goodside
	[response_author]:
		NetOrBrain
	[subreddit]:
		MachineLearning
	[thread_id]:
		5h6yvl
```
