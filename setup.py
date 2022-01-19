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
# Module: This module contains the SETUP for the project
######################################################################

import os

from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="A short description of the project.",
    author="Group_6",
    license="",
)

try:
    os.makedirs("checkpoints/", exist_ok=True)
    print("Directory created successfully")
except OSError as error:
    print("Directory can not be created")
