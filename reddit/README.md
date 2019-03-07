## Create reddit examples

Download a key for a service account with access to Dataflow,
and set `GOOGLE_APPLICATION_CREDENTIALS`:

```
export GOOGLE_APPLICATION_CREDENTIALS={{key location}}
```

Run the Dataflow script.

```
PROJECT="ABCDE"
BUCKET="your-bucket"

DATADIR="gs://${BUCKET}/reddit_$(date +"%Y%m%d")"

# For testing with 10k comments:
TABLE=data.reddit


python data/reddit/create_data.py \
  --output_dir ${DATADIR?} \
  --reddit_table ${PROJECT?}:${TABLE?} \
  --runner DataflowRunner \
  --temp_location ${DATADIR}/temp \
  --staging_location ${DATADIR?}/staging \
  --project ${PROJECT?}
```

View the running job on the
[dataflow admin page](https://console.cloud.google.com/dataflow).

View the final output with:

```
python tools/read_records.py -- \
  --tfrecords_file ${DATADIR?}/test-00000-of-00100.tfrecords | less
```
