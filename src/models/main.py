from train_model import Training_loop

if __name__ == "__main__":
    trainer = Training_loop()
    trainer.loop()




'''
gcloud ai-platform jobs submit training ${JOB_NAME} \
  --region=us-central1 \
  --master-image-uri=gcr.io/cloud-ml-public/training/pytorch-xla.1-10 \
  --scale-tier=BASIC \
  --job-dir=${JOB_DIR} \
  --package-path=./trainer \
  --module-name=trainer.task \
  -- \
  --train-files=gs://cloud-samples-data/ai-platform/chicago_taxi/training/small/taxi_trips_train.csv \
  --eval-files=gs://cloud-samples-data/ai-platform/chicago_taxi/training/small/taxi_trips_eval.csv \
  --num-epochs=10 \
  --batch-size=100 \
  --learning-rate=0.001

'''