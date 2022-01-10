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

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from model_architecture import XrayClassifier
from torch import nn


def get_model_from_checkpoint(path: str) -> nn.Module:
    """Returns a loaded model from checkpoint"""

    from src.models.model_architecture import XrayClassifier

    if not os.path.exists(path):
        raise FileNotFoundError

    model = XrayClassifier()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


@hydra.main(config_name="config.yaml")
def make_predictions(
    config: omegaconf.dictconfig.DictConfig, model: nn.Module = None, load_model: bool = False
) -> None:
    """Classify unseen images from a validation set the model hasn't seen in training or testing"""

    # Load dataset
    dataset = torch.load(config.VALIDATIONSET)

    # Disable gradient tracking
    with torch.no_grad():
        # Retrieve item
        index = 256
        item = dataset[index]
        image = item[0]
        true_target = item[1]

        if load_model:
            # Loading the saved model
            model = get_model_from_checkpoint(config.MODEL_PATH)

        model = XrayClassifier()
        # Generate prediction
        prediction = model(image)

        # Predicted class value using argmax
        predicted_class = np.argmax(prediction)

        # Reshape image
        image = image.reshape(28, 28, 1)

        # Show result
        plt.imshow(image, cmap="gray")
        plt.title(f"Prediction: {predicted_class} - Actual target: {true_target}")
        plt.show()


if __name__ == "__main__":
    make_predictions()
