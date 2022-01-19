#!/usr/bin/env python3
######################################################################
# Authors:      <s202540> Rian Leevinson
#                     <s202385> David Parham
#                     <s193647> Stefan Nahstoll
#                     <s210246> Abhista Partal Balasubramaniam
#
# Course:        Machine Learning Operations
# Semester:    Spring 2022
# Institution:  Technical University of Denmark (DTU)
#
# Module: This module is responsible for prediction
######################################################################

import os
from typing import List

import numpy as np
import torch
from dataset_fetcher import Dataset_fetcher
from cloud_functions import loadCheckpointFromGCP
from matplotlib import pyplot as plt
import omegaconf
from omegaconf import OmegaConf
from torch import nn
from tqdm import tqdm

import wandb


def get_model_from_checkpoint(
    config: omegaconf.dictconfig.DictConfig,
    cloudModel: bool = True
) -> nn.Module:
    """Returns a loaded model from checkpoint"""

    from model_architecture import XrayClassifier

    model = XrayClassifier()
    if cloudModel:
        print("[INFO] Load model from cloud...")
        checkpoint = loadCheckpointFromGCP(config)
    else:
        print("[INFO] Load model from disk...")
        checkpoint = torch.load(config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def create_plot(data: List[torch.tensor], classes: tuple) -> plt:
    """Create plots"""

    # initialize a figure
    fig = plt.figure(figsize=(8, 8))

    # loop over the stored images
    for i, (image, label, prediction) in enumerate(data):

        # create a subplot
        ax = plt.subplot(2, 2, i + 1)

        # grab the image, convert it from channels first ordering to
        # channels last ordering, and scale the raw pixel intensities
        # to te range [0, 255]
        image = image[0].numpy()
        image = (image * 255.0).astype("uint8")
        # show the image along with the label
        plt.imshow(np.transpose(image, (1, 2, 0)))
        plt.title(f"Groundtruth: {classes[label]}", fontsize=14)
        plt.ylabel(f"Predicted: {classes[prediction]}", fontsize=14)
        plt.xticks([])
        plt.yticks([])

    return plt


def inference(model: nn.Module = None, load_model: bool = False) -> None:
    """Classify unseen images from a validation set the model hasn't seen in training or testing"""

    # set flags / seeds
    np.random.seed(1)
    torch.manual_seed(1)

    BASE_DIR = os.getcwd()

    # Load config file
    config = OmegaConf.load(BASE_DIR + "/config/config.yaml")

    # Initialize logging with wandb and track conf settings
    wandb.init(project="MLOps-Project")

    # Optimizer Hyperparameter / const variables
    BATCH_SIZE = 1
    N_WORKERS = config.N_WORKERS

    VAL_PATHS = {
        "images": BASE_DIR + config.TEST_PATHS.images,
        "labels": BASE_DIR + config.TEST_PATHS.labels,
    }

    print("[INFO] Load dataset from disk...")
    validation_set = Dataset_fetcher(VAL_PATHS["images"], VAL_PATHS["labels"])

    print("[INFO] Prepare dataloader...")
    validationloader = torch.utils.data.DataLoader(
        validation_set, shuffle=False, num_workers=N_WORKERS, batch_size=BATCH_SIZE
    )

    classes = ("covid", "normal", "pneumonia")

    if load_model:
        # Loading saved model
        model = get_model_from_checkpoint(config)

    wandb.watch(model, log_freq=100)

    # Disable gradient tracking
    with torch.no_grad():
        model.eval()

        correct = 0
        total = 0

        stored_images = []

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with tqdm(validationloader) as progress_bar:
            for i, (images, labels) in enumerate(progress_bar, 1):

                progress_bar.set_description("[INFO] Running inference...")

                # Generate prediction
                output = model(images)

                # Predicted class value
                _, predicted = torch.max(output.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                stored_images.append((images, labels, predicted))

                if i % 4 == 0:
                    plot = create_plot(stored_images, classes)
                    wandb.log({"examples": wandb.Image(plot)})
                    plot.close()
                    stored_images = []

                # collect the correct predictions for each class
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

    print("[INFO] Calculating model performance...")
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for classes:  {classname} is {accuracy:.1f} %")

    print(
        f"\nAccuracy of the network on the {total} validation images: {100 * correct // total} %\n"
    )


if __name__ == "__main__":

    # this path must be adapted to your own machine
    root_dir = os.getcwd() + "/"  # "/home/davidparham/Workspaces/DTU/MLOps/project/"

    # TODO: Validation files need to be created
    VAL_PATHS = {
        "images": root_dir + "data/preprocessed/covid_not_norm/valid_images.pt",
        "labels": root_dir + "data/preprocessed/covid_not_norm/valid_labels.pt",
    }

    inference(VAL_PATHS, load_model=True)
