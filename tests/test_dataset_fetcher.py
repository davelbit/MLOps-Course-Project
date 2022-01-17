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

# from typing import Union

# from src.models import dataset_fetcher

root_dir = "/home/rianleevinson/MLOps-Course-Project/"
path_img = root_dir + "data/preprocessed/covid_not_norm/train_images.pt"
path_lab = root_dir + "data/preprocessed/covid_not_norm/train_labels.pt"


# def test_dataset_fetcher_images():
#     fetcher = dataset_fetcher.Dataset_fetcher()
#     assert len(fetcher.images) > 0
