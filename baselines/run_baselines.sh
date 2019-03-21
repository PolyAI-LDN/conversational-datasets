#!/bin/bash
set -e
set -o pipefail

if [ "$TFHUB_CACHE_DIR" = "" ]; then
  echo "Set \$TFHUB_CACHE_DIR to run baselines efficiently."
  exit
fi

OUTPUT_FILE=baselines/results.csv
echo "method, train, test, recall_k, accuracy" > ${OUTPUT_FILE}

# reddit os amazon
for DATASET in os amazon; do
  # TF_IDF BM25 USE_SIM USE_MAP
  for METHOD in ELMO_SIM ELMO_MAP USE_LARGE_SIM USE_LARGE_MAP; do

    echo "Running ${METHOD} method on ${DATASET} data."

    python baselines/run_baseline.py  \
      --method "${METHOD?}" \
      --train_dataset "data/${DATASET?}-train*" \
      --test_dataset "data/${DATASET?}-test*" \
      --output_file "${OUTPUT_FILE?}" \
      --eval_num_batches 100

  done

done
