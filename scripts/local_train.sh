#!/bin/bash

echo "Training local ML model"

CURRENT_DATE_TIME="`date +%Y_%m_%d_%H_%M_%S`"
MODEL_NAME=${CURRENT_DATE_TIME} # change to your model name

PACKAGE_PATH=trainer
TRAIN_FILES=data/train.tfrecord
EVAL_FILES=data/eval.tfrecord
MODEL_DIR=trained_models/${MODEL_NAME}


gcloud ml-engine local train \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        -- \
        --train=${TRAIN_FILES} \
        --max-steps=10 \
        --batch-size=1 \
        --eval=${EVAL_FILES} \
        --learning-rate=0.001 \
        --job-dir=${MODEL_DIR} \


ls ${MODEL_DIR}/export/estimator
MODEL_LOCATION=${MODEL_DIR}/export/estimator/$(ls ${MODEL_DIR}/export/estimator | tail -1)
echo ${MODEL_LOCATION}
ls ${MODEL_LOCATION}

# invoke trained model to make prediction given new data instances
gcloud ml-engine local predict \
        --model-dir=${MODEL_LOCATION} \
        --json-instances=data/inference_data.json
