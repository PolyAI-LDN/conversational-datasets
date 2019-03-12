# OpenSubtitles Data

A dataset contains examples of pairs of sentences that occur in sequence in the
data.

## Statistics

* Input files: 4,415
* Number of examples: 320,233,703
* Train set size: 286,655,424
* Test set size: 33,578,279

Typical metrics for the Dataflow job:

* Total vCPU time:  57.657 vCPU hr
* Total memory time: 216.214 GB hr
* Total persistent disk time: 14,414.275 GB hr
* Elapsed time: 25m (225 workers)


## Create a dataset

Download monolingual raw text data,

* English [en.txt.gz](http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.en.gz).


Extract and upload to GCS.

```
gunzip -k en.txt.gz
mkdir lines
split -a 3 -l 100000 en.txt lines/lines-

BUCKET="your-bucket"
gsutil -m cp -r lines gs://${BUCKET?}/opensubtitles/raw/
```


Run the dataflow script:

```
PROJECT="your-google-cloud-project"

DATADIR="gs://${BUCKET?}/opensubtitles/$(date +"%Y%m%d")"

python opensubtitles/create_data.py \
  --output_dir ${DATADIR?} \
  --sentence_files gs://${BUCKET?}/opensubtitles/raw/lines-* \
  --runner DataflowRunner \
  --temp_location ${DATADIR?}/temp \
  --staging_location ${DATADIR?}/staging \
  --project ${PROJECT?}
```

View the running job on the
[dataflow admin page](https://console.cloud.google.com/dataflow).


View final output:

```
python tools/tfrutil.py pp ${DATADIR?}/test-00000-of-00100.tfrecords
```

(It may be faster to copy the tfrecord file locally first.) This will print examples like:

```
[Context]:
	Oh, my God, we killed her.
[Response]:
	Artificial intelligences cannot, by definition, be killed, Dr. Palmer.

Extra Contexts:
	[context/9]:
		So what are we waiting for?
	[context/8]:
		Nothing, it...
	[context/7]:
		It's just if...
	[context/6]:
		If we've underestimated the size of the artifact's data stream...
	[context/5]:
		We'll fry the ship's CPU and we'll all spend the rest of our lives stranded in the Temporal Zone.
	[context/4]:
		The ship's CPU has a name.
	[context/3]:
		Sorry, Gideon.
	[context/2]:
		Can we at least talk about this before you connect...
	[context/1]:
		Gideon?
	[context/0]:
		You still there?

Other features:
	[file_id]:
		lines-emk
```
