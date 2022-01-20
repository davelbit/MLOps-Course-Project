import hiddenlayer as hl
import torch
from model_architecture import XrayClassifier

model = XrayClassifier()

# Build HiddenLayer graph
# Jupyter Notebook renders it automatically
hl_graph = hl.build_graph(model, torch.zeros([1, 1, 512, 512]))

hl_graph.theme = hl.graph.THEMES["blue"].copy()  # Two options: basic and blue
hl_graph.save("cnn_modelviz", format="png")
