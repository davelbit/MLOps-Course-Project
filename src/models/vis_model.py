import hiddenlayer as hl
import torch
from model_architecture import XrayClassifier

model = XrayClassifier()

transforms = [
    # Fold Conv, BN, RELU layers into one
    hl.transforms.Fold("Conv > BatchNorm > Dropout > Relu", "ConvBnDropRelu"),
    hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
    hl.transforms.Fold("Conv > BatchNorm > MaxPool > Relu", "ConvBnMaxPoolRelu"),
    hl.transforms.Fold("Conv > Relu", "ConvRelu"),
    hl.transforms.Prune("Constant"),
    # hl.transforms.Prune('Dropout')
    # Fold repeated blocks
    hl.transforms.FoldDuplicates(),
]
# Build HiddenLayer graph
# Jupyter Notebook renders it automatically
hl_graph = hl.build_graph(model, torch.zeros([1, 1, 512, 512]), transforms=transforms)

hl_graph.theme = hl.graph.THEMES["blue"].copy()  # Two options: basic and blue
hl_graph.save("cnn_modelviz_folded", format="png")
