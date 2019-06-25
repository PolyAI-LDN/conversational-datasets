# OpenSubtitles Data

This dataset uses movie and television subtitles data from OpenSubtitles. The
script in this directory uses the corpus collected by Lison and Tiedemann. See http://opus.nlpl.eu/OpenSubtitles-v2018.php and the following citation:

*OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles.* P. Lison and J. Tiedemann.  In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)

The data is available in 62 different languages.

Consecutive lines in the subtitle data are used to create conversational examples.
There is no guarantee that different lines correspond to different
speakers, but the data nevertheless contains a lot of interesting examples
for modelling the mapping from conversational contexts to responses.

The script filters short and long lines, and strips some text such as
character names and auditory description text.

## Statistics

Below are statistics for the English dataset:

* Input files: 4,415
* Number of examples: 316,891,717
* Train set size: 283,651,561
* Test set size: 33,240,156

Typical metrics for the Dataflow job:

* Total vCPU time:  57.657 vCPU hr
* Total memory time: 216.214 GB hr
* Total persistent disk time: 14,414.275 GB hr
* Elapsed time: 25m (225 workers)
* Estimated cost: 5 USD


# Create the conversational dataset

Below are instructions for creating the conversational dataset from the
OpenSubtitles corpus.

## Download the OpenSubtitles data

First, download monolingual raw text data for the target language.

Visit http://opus.nlpl.eu/OpenSubtitles-v2018.php, and find the *Statistics and TMX/Moses Downloads* table. Click on the language ID in the first column
to get the monolingual plain text file (untokenized).

For English the correct link is:

http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.en.gz

Extract the data, split it into shards, and upload the data to your Google cloud storage bucket:

```bash
gunzip -k en.txt.gz
mkdir lines
split -a 3 -l 100000 en.txt lines/lines-

BUCKET="your-bucket"
gsutil -m cp -r lines gs://${BUCKET?}/opensubtitles/raw/
```

Note that the exact split command is important, as the train/test split is
computed using the file names.

## Run the dataflow script

Now you can run the dataflow script to read the text files and generate
conversational examples:

```bash
PROJECT="your-google-cloud-project"

DATADIR="gs://${BUCKET?}/opensubtitles/$(date +"%Y%m%d")"

python opensubtitles/create_data.py \
  --output_dir ${DATADIR?} \
  --sentence_files gs://${BUCKET?}/opensubtitles/raw/lines/lines-* \
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

The dataset will be saved in the `$DATADIR` directory, as sharded train and test sets- `gs://your-bucket/opensubtitles/YYYYMMDD/train-*-of-01000.tfrecords` and
`gs://your-bucket/opensubtitles/YYYYMMDD/test-*-of-00100.tfrecords`. (Files will be stored as `.json` shards when using `--dataset_format JSON`.)

For Tensorflow format, you can use [`tools/tfrutil.py`](/tools/tfrutil.py) to inspect the files. For example:

```bash
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
