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
# Module: This module is responsible accessing our data
######################################################################

import os
from typing import Union

import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset


# TODO: Write tests for this module
class Dataset_fetcher(Dataset):
    def __init__(
        self,
        PATH_IMG: str,
        PATH_LAB: str,
        transform: Union[transforms.transforms.Compose, None] = None,
    ) -> None:

        self.images = torch.load(PATH_IMG)
        self.labels = torch.load(PATH_LAB).long()
        self.transform = transform

    def __getitem__(self, idx: int) -> Union[torch.tensor, str]:
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.images)


if __name__ == "__main__":

    BASE_DIR = os.getcwd()

    # Load config file
    config = OmegaConf.load(BASE_DIR + "/config/config.yaml")

    TRAIN_PATHS = {
        "images": BASE_DIR + config.TRAIN_PATHS.images,
        "labels": BASE_DIR + config.TRAIN_PATHS.labels,
    }

    dataset = Dataset_fetcher(TRAIN_PATHS["images"], TRAIN_PATHS["labels"])
    dataloader = DataLoader(dataset, shuffle=False, num_workers=4, batch_size=3)
    image, label = next(iter(dataloader))
