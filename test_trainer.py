import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import pytest
import yaml


from nerf_config import get_config

from trainer import train_and_evaluate 


def test_get_config():
    config_path = 'configs/lego.yaml'

    with open(config_path) as file:
        config = yaml.safe_load(file)

    config = get_config(config_path)

    assert config.dataset_name == 'lego'


