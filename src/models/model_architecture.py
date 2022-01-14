#!/usr/bin/env python3
######################################################################
# Authors:      <s202540> Rian Leevinson
#               <s202385> David Parham
#               <s193647> Stefan Nahstoll
#               <s210246> Abhista Partal Balasubramaniam
#
# Course:       Machine Learning Operations
# Semester:     Spring 2022
# Institution:  Technical University of Denmark (DTU)
#
# Module: This module contains the <PLACEHOLDER> model architecture
######################################################################

import torch
from torch import nn
import torch.nn.functional as F

class XrayClassifier(nn.Module):
    """Model Architecture"""

    def __init__(self, num_classes=3):
        super(XrayClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=20)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=48)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=4)

        self.dropout = nn.Dropout(p=0.2)
        
        self.fc1 = nn.Linear(in_features=48*16*16, out_features=16*16)
        self.fc2 = nn.Linear(in_features=16*16, out_features=num_classes)

    def forward(self, x):
        """Forward pass of the model"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(self.relu1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout(self.relu2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.dropout(self.relu3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.dropout(self.relu4(x))
        x = self.pool4(x)
        x = x.view(x.shape[0], 48*16*16)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
