import torch

# from torch.jit import Error
from torch.utils.data import Dataset

# import numpy as np

# import torchvision

# import os


class Dataset_fetcher(Dataset):
    def __init__(self, PATH_IMG, transform=None):
        self.transform = transform

        self.images = torch.load(PATH_IMG)
        print(self.images)
        # self.labels = torch.load(Path_LAB)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.images)
