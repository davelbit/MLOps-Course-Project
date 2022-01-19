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
# Module: This module is responsible for testing the dataset_fetcher
######################################################################

import __init__
from dataset_fetcher import Dataset_fetcher
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

BASE_DIR = __init__.BASE_DIR
config = OmegaConf.load(BASE_DIR + "/config/config.yaml")

TRAIN_PATHS = {
    "images": BASE_DIR + config.TRAIN_PATHS.images,
    "labels": BASE_DIR + config.TRAIN_PATHS.labels,
}

dataset = Dataset_fetcher(TRAIN_PATHS["images"], TRAIN_PATHS["labels"])
dataloader = DataLoader(dataset, shuffle=False, num_workers=4, batch_size=3)
image, label = next(iter(dataloader))
