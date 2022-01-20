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
# Module: This module is used to test the make dataset file
######################################################################

# WORK IN PROGRESS

import PIL
from PIL import Image

from src.data.make_dataset import kornia_preprocess


def test_kornia_preprocess_type():
    image = Image.new("RGB", (512, 512))
    assert type(kornia_preprocess(image) is PIL.Image.Image)
