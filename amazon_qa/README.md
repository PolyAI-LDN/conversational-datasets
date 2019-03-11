# Amazon QA Data


## Statistics


* Input files: 38
* Number of QA dictionaries: 1,569,513
* Number of tuples: 3,322,057
* Number of de-duplicated tuples: 3,286,815
* Train set size: 2,954,149
* Test set size: 332,666

Typical metrics for the Dataflow job:

* Total vCPU time:  0.642 vCPU hr
* Total memory time: 2.408 GB hr
* Total persistent disk time: 160.509 GB hr
* Elapsed time: 15m (4 workers)

# Create the conversational dataset


## Downloading Amazon QA dataset

Download from http://jmcauley.ucsd.edu/data/amazon/qa/.

```
cd amazon_qa
gunzip *

BUCKET="your-bucket"
gsutil -m cp -r * gs://${BUCKET?}/amazon_qa
```

## Creating QA data


Run the Dataflow script:

```
PROJECT="your-google-cloud-project"

DATADIR="gs://${BUCKET?}/amazon_qa/$(date +"%Y%m%d")"

python amazon_qa/create_data.py \
  --file_pattern gs://${BUCKET}/amazon_qa/raw/* \
  --output_dir ${DATADIR} \
  --runner DataflowRunner --temp_location ${DATADIR}/temp \
  --staging_location ${DATADIR}/staging \
  --project ${PROJECT?}
```

View the running job on the
[dataflow admin page](https://console.cloud.google.com/dataflow).


View final output:

```
please run data:read_records -- \
  --tfrecords_file ${DATADIR?}/test-00000-of-00010.tfrecords \
  --proto_format TF_EXAMPLE | less
```

```
python tools/tfrutil.py pp ${DATADIR?}/test-00000-of-00010.tfrecords
```

(It may be faster to copy the tfrecord file locally first.) This will print examples like:

```

```
