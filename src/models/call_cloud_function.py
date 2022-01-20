import json
import numpy as np
import torch
from torch import nn

import requests


img = np.random.normal(0, 1, [512, 512])

x = {"input_data": img.tolist(), "model-id": "models/D19012022T170140best_model.pth"}

y = json.dumps(x)

request_json = json.loads(y)
data = torch.tensor(request_json["input_data"])

url = "https://europe-west1-charged-city-337910.cloudfunctions.net/COVID"

r = requests.post(url, json=y)
print(r.text)


from google.cloud import storage
import torch
from torch import nn
from omegaconf import OmegaConf
import omegaconf
import io
from datetime import datetime


def loadCheckpointFromGCP(config):
    BUCKET_NAME = config.BUCKET_NAME
    MODEL_FILE = config.BUCKET_BEST_MODEL

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(MODEL_FILE)
    checkpoint = torch.load(io.BytesIO(blob.download_as_string()))
    return checkpoint


def get_model_from_checkpoint(
    config: omegaconf.dictconfig.DictConfig, cloudModel: bool = True
) -> nn.Module:
    """Returns a loaded model from checkpoint"""

    from model_architecture import XrayClassifier

    model = XrayClassifier()
    if cloudModel:
        print("[INFO] Load model from cloud...")
        checkpoint = loadCheckpointFromGCP(config)
    else:
        print("[INFO] Load model from disk...")
        checkpoint = torch.load(config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def predict_covid(request):
    config = OmegaConf.load("config/config.yaml")
    request_json = json.loads(request)
    if request_json and "input_data" in request_json:
        if "model-id" in request_json:
            config.BUCKET_BEST_MODEL = request_json["model-id"]

        model = get_model_from_checkpoint(config)

        data = torch.tensor(request_json["input_data"])
        assert data.shape[-2:] == torch.Size([512, 512])
        data = data.view(-1, 1, 512, 512)
        prediction = model(data)
        print(prediction)


predict_covid(y)
