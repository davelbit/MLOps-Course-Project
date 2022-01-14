

import argparse

import numpy as np
import matplotlib.pyplot as plt
from google.cloud import storage
import os
import torch

"""
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
"""


def run():
    parser = argparse.ArgumentParser(description="model running arguments")
    parser.add_argument(
        "-e",
        "--num-epochs",
        type=int,
        default=5,
        help="number of epochs",
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.01,
        help="learning rate",
    )

    parser.add_argument(
        "-img",
        '--image-file',
        type=str,
        default="data/preprocessed/covid_not_norm/train_images.pt",
        help="location of train tensor",
    )

    parser.add_argument(
        "-lab",
        '--label-file',
        type=str,
        default="data/preprocessed/covid_not_norm/train_labels.pt",
        help="location of train tensor",
    )

    parser.add_argument(
        "-bs",
        '--batch-size',
        type=str,
        default="data/preprocessed/covid_not_norm",
        help="location of train tensor",
    )

    args = parser.parse_args()

    x,y= np.random.randint(0,100,20),np.random.randint(0,1000,20)

    imgs=torch.load(args.image_file)
    labels=torch.load(args.label_file)
    print(imgs[0,0,:,:].shape)
    plt.figure()
    plt.imshow(imgs[0,0,:,:],cmap='gray')
    plt.title(labels[0])
    plt.show()

if __name__ == "__main__":
    run()
