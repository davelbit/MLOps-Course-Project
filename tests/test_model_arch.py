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
# Module: This module is responsible for testing the model architecture
######################################################################

from src.models import model_architecture


def test_model_arch():
    """tests the model architecture"""
    assert model_architecture.XrayClassifier()


# TODO: Needs to be implemented by passing a test image
def test_forward_pass():
    """tests the forward pass"""
    assert model_architecture.forward()
