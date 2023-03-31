import pytest
import yaml


from utils import get_config


def test_get_config():
    config_path = 'configs/lego.yaml'

    with open(config_path) as file:
        config = yaml.safe_load(file)

    config = get_config(config_path)

    assert config.dataset_name == 'lego'


