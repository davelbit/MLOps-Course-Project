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

from typing import Union

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


# TODO: Write tests for this module
class Dataset_fetcher(Dataset):
    def __init__(
        self, PATH_IMG: str, transform: Union[transforms.transforms.Compose, None] = None
    ) -> None:

        print(type(transform))
        self.transform = transform
        self.images = torch.load(PATH_IMG)
        # self.labels = torch.load(Path_LAB)

    def __getitem__(self, idx: int) -> Union[torch.tensor, str]:
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.images)


if __name__ == "__main__":
    path = "data/raw/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset"
    dataset = Dataset_fetcher(path)
    dataloader = DataLoader(dataset, shuffle=False, num_workers=4, batch_size=3)
    image, label = next(iter(dataloader))
