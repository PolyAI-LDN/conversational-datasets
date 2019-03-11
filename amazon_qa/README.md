# Amazon QA Data


## Statistics


* Input files: 38
* Number of QA dictionaries: 

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
