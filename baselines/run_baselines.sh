#!/bin/bash
set -e
set -o pipefail

if [ "$TFHUB_CACHE_DIR" = "" ]; then
  echo "Set \$TFHUB_CACHE_DIR to run baselines efficiently."
  exit
fi

OUTPUT_FILE=baselines/results.csv
echo "method, train, test, recall_k, accuracy" > ${OUTPUT_FILE}

for DATASET in reddit os amazon; do
  for METHOD in TF_IDF BM25 USE_SIM USE_MAP; do

    echo "Running ${METHOD} method on ${DATASET} data."

    python baselines/run_baseline.py  \
      --method "${METHOD?}" \
      --train_dataset "data/${DATASET?}-train*" \
      --test_dataset "data/${DATASET?}-test*" \
      --output_file "${OUTPUT_FILE?}"

  done

done
