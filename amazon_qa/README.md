# Amazon QA Data

This dataset is based on a corpus extracted by McAuley et al., who scraped questions and answers from Amazon. The dataset is described at http://jmcauley.ucsd.edu/data/amazon/qa/ as well as in the following papers:

*Modeling ambiguity, subjectivity, and diverging viewpoints in opinion question answering systems*. Mengting Wan, Julian McAuley. International Conference on Data Mining (ICDM), 2016. [pdf](http://cseweb.ucsd.edu/~jmcauley/pdfs/icdm16c.pdf)

*Addressing complex and subjective product-related queries with customer reviews*. Julian McAuley, Alex Yang. World Wide Web (WWW), 2016. [pdf](http://cseweb.ucsd.edu/~jmcauley/pdfs/www16b.pdf)

The script in this directory processes this corpus, filters long and short texts, and creates a conversational dataset.

## Statistics

Below are some statistics of the conversational dataset:

* Input files: 38
* Number of QA dictionaries: 1,569,513
* Number of tuples: 4,035,625
* Number of de-duplicated tuples: 3,689,912
* Train set size: 3,316,905
* Test set size: 373,007

Typical metrics for the Dataflow job:

* Total vCPU time:  0.707 vCPU hr
* Total memory time: 2.653 GB hr
* Total persistent disk time: 176.864 GB hr
* Elapsed time: 17m (4 workers)
* Estimated cost: less than 10 cents

# Create the conversational dataset

Below are instructions for how to generate the Amazon QA conversational dataset.

## Downloading Amazon QA dataset

First you must download the input data from http://jmcauley.ucsd.edu/data/amazon/qa/. In total there are 38 `.json.gz` files to download. Unzip them all and copy them to your Google cloud storage bucket:

```bash
gunzip *

BUCKET="your-bucket"
gsutil -m cp -r * gs://${BUCKET?}/amazon_qa/raw/
```

Note that while the files are named `.json`, they are not actually valid
JSON, but rather python dictionaries in string format.

## Run the dataflow script

Run the following command to process the raw input data into a conversational
dataset:

```bash
PROJECT="your-google-cloud-project"

DATADIR="gs://${BUCKET?}/amazon_qa/$(date +"%Y%m%d")"

python amazon_qa/create_data.py \
  --file_pattern gs://${BUCKET?}/amazon_qa/raw/* \
  --output_dir ${DATADIR} \
  --runner DataflowRunner --temp_location ${DATADIR}/temp \
  --staging_location ${DATADIR}/staging \
  --project ${PROJECT?} \
  --dataset_format TF
```
You may use `--dataset_format JSON` to output JSON examples, rather than serialized Tensorflow examples in TFRecords.

Once the above is running, you can continue to monitor it in the terminal, or quit the process and follow the running job on the
[dataflow admin page](https://console.cloud.google.com/dataflow).

Please confirm that the statistics reported on the dataflow job page agree with the statistics reported above, to ensure you have a correct version of the dataset.

The dataset will be saved in the `$DATADIR` directory, as sharded train and test sets- `gs://your-bucket/amazon_qa/YYYYMMDD/train-*-of-00100.tfrecord` and
`gs://your-bucket/amazon_qa/YYYYMMDD/test-*-of-00010.tfrecord`. (Files will be stored as `.json` shards when using `--dataset_format JSON`.)

For Tensorflow format, you can use [`tools/tfrutil.py`](/tools/tfrutil.py) to inspect the files. For example:

```bash
python tools/tfrutil.py pp ${DATADIR?}/test-00000-of-00010.tfrecord
```

(It may be faster to copy the tfrecord file locally first.) This will print examples like:

```
Example 0
--------
[Context]:
        can I use this filter in Australian 220V power?
[Response]:
        No it is a 110 unit you would have to purchase a 110 converter elseware to use this

Other features:
        [product_id]:
                B002WJ34IC
--------


Example 1
--------
[Context]:
        can 11 year olds ride it
[Response]:
        this car is too small for 11 year olds, I would prefer age between 4 - 6 for this car.

Other features:
        [product_id]:
                B00E0GWJY0
--------
```
