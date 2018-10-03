echo "Submitting a Cloud ML Engine job..."

REGION="us-east1"
TIER="BASIC_GPU" # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
BUCKET="joelsdata" # change to your bucket name

CURRENT_DATE_TIME="`date +%Y_%m_%d_%H_%M_%S`"
MODEL_NAME=${CURRENT_DATE_TIME}

PACKAGE_PATH=trainer # this can be a gcs location to a zipped and uploaded package
TRAIN_FILES=gs://${BUCKET}/synthetic_data/train.tfrecord
EVAL_FILES=gs://${BUCKET}/synthetic_data/eval.tfrecord
MODEL_DIR=gs://${BUCKET}/model_dir/${MODEL_NAME}

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${MODEL_NAME}_${TIER}_${CURRENT_DATE}
#JOB_NAME=tune_${MODEL_NAME}_${CURRENT_DATE} # for hyper-parameter tuning jobs

gcloud ml-engine jobs submit training ${JOB_NAME} \
        --job-dir=${MODEL_DIR} \
        --region=${REGION} \
        --scale-tier=${TIER} \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH}  \
        --runtime-version=1.10 \
        --config=config.yaml \
        -- \
        --train=${TRAIN_FILES} \
        --max-steps=10000 \
        --batch-size=16 \
        --eval=${EVAL_FILES} \



echo "To view tensorboard type"
echo "tensorboard --logdir=$MODEL_DIR"
# notes:
# use --packages instead of --package-path if gcs location
# add --reuse-job-dir to resume training