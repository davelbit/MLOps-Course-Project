from google.cloud import storage
import torch
from torch import nn
import omegaconf
from omegaconf import OmegaConf
import io
import functions_framework
import numpy as np

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


@functions_framework.http
def predict_covid(request):
    config = OmegaConf.load("config.yaml")

    request_json=request.get_json()#json.loads(request)
    if request_json and 'input_data' in request_json:
        if 'model-id' in request_json:
            config.BUCKET_BEST_MODEL=request_json['model-id']


        data = torch.tensor(request_json['input_data'])
        
        if not data.shape[-2:]==torch.Size([512, 512]):
            return 'Wrong size of image, should be 512x512'
        data=data.view(-1,1,512,512)

        model = get_model_from_checkpoint(config)
        prediction=model(data).data
        diagnosis=['Covid','Normal','Pneumonia']

        return f'Diagnosis: {diagnosis[np.argmax(prediction.numpy())]}'
    else:
        return 'No input data received'
