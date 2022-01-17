import hydra
from torch import nn

from src.models.predict_model import get_model_from_checkpoint


@hydra.main(config_name="test_config.yaml")
def test_get_model_from_checkpoint(config):
    assert isinstance(get_model_from_checkpoint(config.MODEL_PATH), nn.Module)
