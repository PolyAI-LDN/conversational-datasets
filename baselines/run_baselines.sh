#!/bin/bash
set -e
set -o pipefail

if [ "$TFHUB_CACHE_DIR" = "" ]; then
  echo "Set \$TFHUB_CACHE_DIR to run baselines efficiently."
  exit
fi

OUTPUT_FILE=baselines/results.csv
echo "method, train, test, train_size, test_size, recall_k, accuracy" > ${OUTPUT_FILE}

for DATASET in reddit os amazon; do
  for METHOD in TF_IDF BM25 USE_SIM USE_MAP USE_LARGE_SIM USE_LARGE_MAP ELMO_SIM ELMO_MAP BERT_SMALL_SIM BERT_SMALL_MAP BERT_LARGE_SIM BERT_LARGE_MAP USE_QA_SIM USE_QA_MAP; do

    echo "Running ${METHOD} method on ${DATASET} data."

    EVAL_NUM_BATCHES=500
    if [[ "$METHOD" == USE_LARGE*  ]] || [[ "$METHOD" == ELMO*  ]] || [[ "$METHOD" == BERT*  ]] || [[ "$METHOD" == USE_QA* ]]; then
      # These models are slow, so reduce the number of eval batches.
      EVAL_NUM_BATCHES=100
    fi

    python baselines/run_baseline.py  \
      --method "${METHOD?}" \
      --train_dataset "data/${DATASET?}-train-sampled" \
      --test_dataset "data/${DATASET?}-test-sampled" \
      --output_file "${OUTPUT_FILE?}" \
      --eval_num_batches "${EVAL_NUM_BATCHES?}"

  done

done
